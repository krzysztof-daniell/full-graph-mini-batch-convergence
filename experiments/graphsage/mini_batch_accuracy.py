import argparse
from collections.abc import Callable
from timeit import default_timer
from typing import Union

import dgl
import sigopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GraphSAGE
from utils import (Callback, download_dataset, log_metrics_to_sigopt,
                   process_dataset)


def train(
    model: nn.Module,
    device: Union[str, torch.device],
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dataloader: dgl.dataloading.NodeDataLoader,
) -> tuple[float]:
    model.train()

    total_loss = 0
    total_accuracy = 0

    start = default_timer()

    for step, (_, _, blocks) in enumerate(dataloader):
        optimizer.zero_grad()

        blocks = [block.int().to(device) for block in blocks]

        inputs = blocks[0].srcdata['feat']
        labels = blocks[-1].dstdata['label']

        logits = model(blocks, inputs)
        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()

        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        accuracy = correct.item() / len(labels)

        total_loss += loss.item()
        total_accuracy += accuracy

    stop = default_timer()
    time = stop - start

    total_loss /= step + 1
    total_accuracy /= step + 1

    return time, total_loss, total_accuracy


def validate(
    model: nn.Module,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    g: dgl.DGLGraph,
    mask: torch.Tensor,
) -> tuple[float]:
    features = g.ndata['feat']
    labels = g.ndata['label']

    model.eval()

    start = default_timer()

    with torch.no_grad():
        logits = model(g, features)
        loss = loss_function(logits[mask], labels[mask])

        _, indices = torch.max(logits[mask], dim=1)
        correct = torch.sum(indices == labels[mask])
        accuracy = correct.item() / len(labels[mask])

    stop = default_timer()
    time = stop - start

    return time, loss, accuracy


def run(args: argparse.ArgumentParser) -> None:
    torch.manual_seed(args.seed)

    dataset = process_dataset(args.dataset, '/home/ksadowski/datasets')
    g = dataset[0]

    train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    valid_idx = torch.nonzero(g.ndata['valid_mask'], as_tuple=True)[0]
    test_idx = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    aggregator_types = {
        'mean': 0,
        'gcn': 1,
        'lstm': 2,
    }
    activations = {
        'relu': 0,
        'leaky_relu': 1,
    }

    sigopt.params.setdefaults({
        'lr': args.lr,
        'hidden_feats': args.hidden_feats,
        'num_layers': args.num_layers,
        'aggregator_type': aggregator_types[args.aggregator_type],
        'batch_norm': int(args.batch_norm),
        'activation': activations[args.activation],
        'input_dropout': args.input_dropout,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
    })

    fanouts = [int(i) for i in args.fanouts.split(',')]

    for i in reversed(range(len(fanouts))):
        sigopt.params.setdefaults({f'layer_{i + 1}_fanout': fanouts[i]})

        fanouts.pop(i)

    for i in range(sigopt.params.num_layers):
        fanouts.append(sigopt.params[f'layer_{i + 1}_fanout'])

    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts)
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_idx,
        sampler,
        batch_size=sigopt.params.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
    )

    # train_dataloader = dgl.dataloading.NodeDataLoader(
    #     g,
    #     train_idx,
    #     sampler,
    #     batch_size=sigopt.params.batch_size * 256,  # int range(1, 128)
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=4,
    # )

    in_feats = g.ndata['feat'].shape[-1]
    out_feats = dataset.num_classes

    aggregator_types = {
        '0': 'mean',
        '1': 'gcn',
        '2': 'lstm',
    }
    activations = {
        '0': F.relu,
        '1': F.leaky_relu,
    }

    model = GraphSAGE(
        in_feats,
        sigopt.params.hidden_feats,
        out_feats,
        sigopt.params.num_layers,
        aggregator_type=aggregator_types[f'{sigopt.params.aggregator_type}'],
        batch_norm=bool(sigopt.params.batch_norm),
        input_dropout=sigopt.params.input_dropout,
        dropout=sigopt.params.dropout,
        activation=activations[f'{sigopt.params.activation}'],
    ).to(device)

    # model = GraphSAGE(
    #     in_feats,
    #     sigopt.params.hidden_feats * 16,  # int range(4, 64)
    #     out_feats,
    #     sigopt.params.num_layers,  # int range(2, 5)
    #     aggregator_types[f'{sigopt.params.aggregator_type}'],
    #     bool(sigopt.params.batch_norm),
    #     activations[f'{sigopt.params.activation}'],
    #     sigopt.params.input_dropout * 0.1,  # int range(0, 9)
    #     sigopt.params.dropout * 0.1,  # int range(0, 9)
    # ).to(device)

    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=sigopt.params.lr)

    checkpoint = Callback(args.early_stopping_patience,
                          args.early_stopping_monitor)

    for epoch in range(args.num_epochs):
        train_time, train_loss, train_accuracy = train(
            model, device, optimizer, loss_function, train_dataloader)
        valid_time, valid_loss, valid_accuracy = validate(
            model, loss_function, g, valid_idx)

        checkpoint.create(
            epoch,
            train_time,
            valid_time,
            train_loss,
            valid_loss,
            train_accuracy,
            valid_accuracy,
            model,
        )

        print(
            f'Epoch: {epoch + 1:03} '
            f'Train Loss: {train_loss:.2f} '
            f'Valid Loss: {valid_loss:.2f} '
            f'Train Accuracy: {train_accuracy * 100:.2f} % '
            f'Valid Accuracy: {valid_accuracy * 100:.2f} % '
            f'Train Epoch Time: {train_time:.2f} '
            f'Valid Epoch Time: {valid_loss:.2f} '
        )

        if checkpoint.should_stop:
            print('!! Early Stopping !!')

            break

    model.load_state_dict(checkpoint.best_epoch_model_parameters)

    test_time, test_loss, test_accuracy = validate(
        model, loss_function, g, test_idx)

    print(
        f'Test Loss: {test_loss:.2f} '
        f'Test Accuracy: {test_accuracy * 100:.2f} % '
        f'Test Epoch Time: {test_time:.2f}'
    )

    log_metrics_to_sigopt(
        checkpoint,
        'GraphSAGE NS',
        args.dataset,
        test_loss,
        test_accuracy,
        test_time,
    )


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GraphSAGE Accuracy Optimization')

    argparser.add_argument('--dataset', default='ogbn-products', type=str)
    argparser.add_argument('--download-dataset', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--num-epochs', default=500, type=int)
    argparser.add_argument('--lr', default=0.003, type=float)
    argparser.add_argument('--hidden-feats', default=256, type=int)
    argparser.add_argument('--num-layers', default=3, type=int)
    argparser.add_argument('--aggregator-type', default='mean',
                           type=str, choices=['mean', 'gcn', 'lstm'])
    argparser.add_argument('--batch-norm', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--activation', default='relu',
                           type=str, choices=['relu', 'leaky_relu'])
    argparser.add_argument('--input-dropout', default=0, type=float)
    argparser.add_argument('--dropout', default=0.5, type=float)
    argparser.add_argument('--batch-size', default=1000, type=int)
    argparser.add_argument('--fanouts', default='5,10,15', type=str)
    argparser.add_argument('--seed', default=13, type=int)
    argparser.add_argument('--early-stopping-patience', default=10, type=int)
    argparser.add_argument('--early-stopping-monitor',
                           default='loss', type=str)

    args = argparser.parse_args()

    if args.download_dataset:
        download_dataset(args.dataset)

    run(args)
