import argparse
from collections.abc import Callable
from timeit import default_timer
from typing import Union

import dgl
import sigopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GAT
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
    inputs = g.ndata['feat']
    labels = g.ndata['label']

    model.eval()

    start = default_timer()

    with torch.no_grad():
        logits = model(g, inputs)
        loss = loss_function(logits[mask], labels[mask])

        _, indices = torch.max(logits[mask], dim=1)
        correct = torch.sum(indices == labels[mask])
        accuracy = correct.item() / len(labels[mask])

    stop = default_timer()
    time = stop - start

    return time, loss, accuracy


def run(args: argparse.ArgumentParser) -> None:
    torch.manual_seed(args.seed)

    dataset, g, train_idx, valid_idx, test_idx = process_dataset(
        args.dataset,
        root='/home/ksadowski/datasets',
        reverse_edges=args.graph_reverse_edges,
        self_loop=args.graph_self_loop,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    norms = {'both': 0, 'left': 1, 'none': 2, 'right': 3}
    activations = {'leaky_relu': 0, 'relu': 1}

    sigopt.params.setdefaults({
        'lr': args.lr,
        'hidden_feats': args.hidden_feats,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'norm': norms[args.norm],
        'batch_norm': int(args.batch_norm),
        'input_dropout': args.input_dropout,
        'attn_dropout': args.attn_dropout,
        'edge_dropout': args.edge_dropout,
        'dropout': args.dropout,
        'negative_slope': args.negative_slope,
        'residual': int(args.residual),
        'activation': activations[args.activation],
        'use_attn_dst': int(args.use_attn_dst),
        'bias': int(args.bias),
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

    node_in_feats = g.ndata['feat'].shape[-1]
    edge_in_feats = 0
    out_feats = dataset.num_classes

    norms = {'0': 'both', '1': 'left', '2': 'none', '3': 'right'}
    activations = {'0': F.leaky_relu, '1': F.relu}

    model = GAT(
        node_in_feats,
        edge_in_feats,
        sigopt.params.hidden_feats,
        out_feats,
        sigopt.params.num_heads,
        sigopt.params.num_layers,
        norm=norms[f'{sigopt.params.norm}'],
        batch_norm=bool(sigopt.params.batch_norm),
        input_dropout=sigopt.params.input_dropout,
        attn_dropout=sigopt.params.attn_dropout,
        edge_dropout=sigopt.params.edge_dropout,
        dropout=sigopt.params.dropout,
        negative_slope=sigopt.params.negative_slope,
        residual=bool(sigopt.params.residual),
        activation=activations[f'{sigopt.params.activation}'],
        use_attn_dst=bool(sigopt.params.use_attn_dst),
        bias=bool(sigopt.params.bias),
    ).to(device)

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
            f'Valid Epoch Time: {valid_loss:.2f}'
        )

        if checkpoint.should_stop:
            print('!! Early Stopping !!')

            break

    if args.test_validation:
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
            'GAT NS',
            args.dataset,
            test_loss,
            test_accuracy,
            test_time,
        )
    else:
        log_metrics_to_sigopt(checkpoint, 'GAT NS', args.dataset)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GAT NS Optimization')

    argparser.add_argument('--dataset', default='ogbn-products', type=str,
                           choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins'])
    argparser.add_argument('--download-dataset', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--graph-reverse-edges', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--graph-self-loop', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--num-epochs', default=500, type=int)
    argparser.add_argument('--lr', default=0.001, type=float)
    argparser.add_argument('--hidden-feats', default=128, type=int)
    argparser.add_argument('--num-heads', default=4, type=int)
    argparser.add_argument('--num-layers', default=3, type=int)
    argparser.add_argument('--norm', default='none',
                           type=str, choices=['both', 'left', 'none', 'right'])
    argparser.add_argument('--batch-norm', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--input-dropout', default=0, type=float)
    argparser.add_argument('--attn-dropout', default=0, type=float)
    argparser.add_argument('--edge-dropout', default=0, type=float)
    argparser.add_argument('--dropout', default=0, type=float)
    argparser.add_argument('--negative-slope', default=0.2, type=float)
    argparser.add_argument('--residual', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--activation', default='relu',
                           type=str, choices=['leaky_relu', 'relu'])
    argparser.add_argument('--use-attn-dst', default=True,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--bias', default=True,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--batch-size', default=512, type=int)
    argparser.add_argument('--fanouts', default='10,10,10', type=str)
    argparser.add_argument('--early-stopping-patience', default=10, type=int)
    argparser.add_argument('--early-stopping-monitor',
                           default='loss', type=str)
    argparser.add_argument('--test-validation', default=True,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--seed', default=13, type=int)

    args = argparser.parse_args()

    if args.download_dataset:
        download_dataset(args.dataset)

    run(args)
