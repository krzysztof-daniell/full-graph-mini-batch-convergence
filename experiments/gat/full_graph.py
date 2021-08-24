import argparse
from collections.abc import Callable
from timeit import default_timer

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
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    g: dgl.DGLGraph,
    mask: torch.Tensor,
) -> tuple[float]:
    inputs = g.ndata['feat']
    labels = g.ndata['label']

    model.train()
    optimizer.zero_grad()

    start = default_timer()

    logits = model(g, inputs)
    loss = loss_function(logits[mask], labels[mask])

    loss.backward()
    optimizer.step()

    loss = loss.item()

    _, indices = torch.max(logits[mask], dim=1)
    correct = torch.sum(indices == labels[mask])
    accuracy = correct.item() / len(labels[mask])

    stop = default_timer()
    time = stop - start

    return time, loss, accuracy

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

    sigopt.params.setdefaults({
        'lr': args.lr,
        'hidden_feats': args.hidden_feats,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'norm': args.norm,
        'batch_norm': int(args.batch_norm),
        'input_dropout': args.input_dropout,
        'attn_dropout': args.attn_dropout,
        'edge_dropout': args.edge_dropout,
        'dropout': args.dropout,
        'negative_slope': args.negative_slope,
        'residual': int(args.residual),
        'activation': args.activation,
        'use_attn_dst': int(args.use_attn_dst),
        'bias': int(args.bias),
    })

    node_in_feats = g.ndata['feat'].shape[-1]
    edge_in_feats = 0
    out_feats = dataset.num_classes

    #norms = {'both': 'both', '1': 'left', '2': 'none', '3': 'right'}
    activations = {'leaky_relu': F.leaky_relu, 'relu': F.relu}

    model = GAT(
        node_in_feats,
        edge_in_feats,
        sigopt.params.hidden_feats,
        out_feats,
        sigopt.params.num_heads,
        sigopt.params.num_layers,
        norm=sigopt.params.norm,
        batch_norm=bool(sigopt.params.batch_norm),
        input_dropout=sigopt.params.input_dropout,
        attn_dropout=sigopt.params.attn_dropout,
        edge_dropout=sigopt.params.edge_dropout,
        dropout=sigopt.params.dropout,
        negative_slope=sigopt.params.negative_slope,
        residual=bool(sigopt.params.residual),
        activation=sigopt.params.activation,
        use_attn_dst=bool(sigopt.params.use_attn_dst),
        bias=bool(sigopt.params.bias),
    ).to(device)

    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=sigopt.params.lr)

    checkpoint = Callback(args.early_stopping_patience,
                          args.early_stopping_monitor)

    for epoch in range(args.num_epochs):
        train_time, train_loss, train_accuracy = train(
            model, optimizer, loss_function, g, train_idx)
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
            'GAT',
            args.dataset,
            test_loss,
            test_accuracy,
            test_time,
        )
    else:
        log_metrics_to_sigopt(checkpoint, 'GAT', args.dataset)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GAT Optimization')
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
