import argparse
import os
from timeit import default_timer
from typing import Callable

import dgl
import sigopt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator

import utils
from model import GAT

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    evaluator: Evaluator,
    g: dgl.DGLGraph,
    mask: torch.Tensor,
) -> tuple[float]:
    model.train()
    optimizer.zero_grad()

    start = default_timer()

    inputs = g.ndata['feat']
    labels = g.ndata['label'][mask]

    logits = model(g, inputs)[mask]

    loss = loss_function(logits, labels)
    score = utils.get_evaluation_score(evaluator, logits, labels)

    loss.backward()
    optimizer.step()

    stop = default_timer()
    time = stop - start

    loss = loss.item()

    return time, loss, score

def validate(
    model: nn.Module,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    evaluator: Evaluator,
    g: dgl.DGLGraph,
    mask: torch.Tensor,
) -> tuple[float]:
    model.eval()

    start = default_timer()

    inputs = g.ndata['feat']
    labels = g.ndata['label'][mask]

    with torch.no_grad():
        logits = model(g, inputs)[mask]

        loss = loss_function(logits, labels)
        score = utils.get_evaluation_score(evaluator, logits, labels)

    stop = default_timer()
    time = stop - start

    loss = loss.item()

    return time, loss, score


def run(
    args: argparse.ArgumentParser, 
    sigopt_context: sigopt.run_context = None,
) -> None:

    torch.manual_seed(args.seed)

    dataset, evaluator, g, train_idx, valid_idx, test_idx = utils.process_dataset(
        args.dataset,
        root=args.dataset_root,
        reverse_edges=args.graph_reverse_edges,
        self_loop=args.graph_self_loop,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if sigopt_context is not None:
        lr = sigopt_context.params.lr
        node_hidden_feats = sigopt_context.params.node_hidden_feats
        num_heads = sigopt_context.params.num_heads
        num_layers = sigopt_context.params.num_layers
        norm = args.norm
        batch_norm = bool(sigopt_context.params.batch_norm)
        input_dropout = sigopt_context.params.input_dropout
        attn_dropout = sigopt_context.params.attn_dropout
        edge_dropout = sigopt_context.params.edge_dropout
        dropout = sigopt_context.params.dropout
        negative_slope = sigopt_context.params.negative_slope
        residual = bool(sigopt_context.params.residual)
        activation = sigopt_context.params.activation
        use_attn_dst = bool(sigopt_context.params.use_attn_dst)
        bias = bool(sigopt_context.params.bias)
    else:
        lr = args.lr
        node_hidden_feats = args.node_hidden_feats
        num_heads = args.num_heads
        num_layers = args.num_layers
        norm = args.norm
        batch_norm = args.batch_norm
        input_dropout = args.input_dropout
        attn_dropout = args.attn_dropout
        edge_dropout = args.edge_dropout
        dropout = args.dropout
        negative_slope = args.negative_slope
        residual = args.residual
        activation = args.activation
        use_attn_dst = args.use_attn_dst
        bias = args.bias
        
    node_in_feats = g.ndata['feat'].shape[-1]

    if args.dataset == 'ogbn-proteins':

        if sigopt_context is not None:
            edge_hidden_feats = sigopt_context.params.edge_hidden_feats
        else:
            edge_hidden_feats = args.edge_hidden_feats

        edge_in_feats = g.edata['feat'].shape[-1]
        
    else:
        edge_in_feats = 0
        edge_hidden_feats = 0

    out_feats = dataset.num_classes

    activations = {'leaky_relu': F.leaky_relu, 'relu': F.relu}

    model = GAT(
        node_in_feats,
        edge_in_feats,
        node_hidden_feats,
        edge_hidden_feats,
        out_feats,
        num_heads,
        num_layers,
        norm=norm,
        batch_norm=batch_norm,
        input_dropout=input_dropout,
        attn_dropout=attn_dropout,
        edge_dropout=edge_dropout,
        dropout=dropout,
        negative_slope=negative_slope,
        residual=residual,
        activation=activations[activation],
        use_attn_dst=use_attn_dst,
        bias=bias,
    ).to(device)

    if args.dataset == 'ogbn-proteins':
        loss_function = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_function = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint = utils.Callback(args.early_stopping_patience,
                                args.early_stopping_monitor)

    for epoch in range(args.num_epochs):
        train_time, train_loss, train_score = train(
            model, optimizer, loss_function, evaluator, g, train_idx)
        valid_time, valid_loss, valid_score = validate(
            model, loss_function, evaluator, g, valid_idx)

        checkpoint.create(
            epoch,
            train_time,
            valid_time,
            train_loss,
            valid_loss,
            train_score,
            valid_score,
            model,
            sigopt_context=sigopt_context
        )

        print(
            f'Epoch: {epoch + 1:03} '
            f'Train Loss: {train_loss:.2f} '
            f'Valid Loss: {valid_loss:.2f} '
            f'Train Score: {train_score:.4f} '
            f'Valid Score: {valid_score:.4f} '
            f'Train Epoch Time: {train_time:.2f} '
            f'Valid Epoch Time: {valid_time:.2f}'
        )

        if checkpoint.should_stop:
            print('!! Early Stopping !!')

            break

    if args.test_validation:

        model.load_state_dict(checkpoint.best_epoch_model_parameters)
        
        test_time, test_loss, test_score = validate(
            model, 
            loss_function, 
            evaluator, 
            g, 
            test_idx
        )

        print(
            f'Test Loss: {test_loss:.2f} '
            f'Test Score: {test_score * 100:.2f} % '
            f'Test Epoch Time: {test_time:.2f}'
        )

    if sigopt_context is not None:

        metrics = {
            'best epoch': checkpoint.best_epoch,
            'best epoch - train loss': checkpoint.best_epoch_train_loss,
            'best epoch - train score': checkpoint.best_epoch_train_accuracy,
            'best epoch - valid loss': checkpoint.best_epoch_valid_loss,
            'best epoch - valid score': checkpoint.best_epoch_valid_accuracy,
            'best epoch - training time': checkpoint.best_epoch_training_time,
            'avg train epoch time': np.mean(checkpoint.train_times),
            'avg valid epoch time': np.mean(checkpoint.valid_times),
            'best epoch - test loss': test_loss,
            'best epoch - test score': test_score,
            'test epoch time': test_time
        }

        utils.log_metrics_to_sigopt(sigopt_context, **metrics)
        
        sigopt_context.end()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GAT NS Optimization')

    argparser.add_argument('--dataset', default='ogbn-products', type=str,
                           choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins'])
    argparser.add_argument('--dataset-root', default='dataset', type=str)
    argparser.add_argument('--download-dataset', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--sigopt-api-token', default=None, type=str)
    argparser.add_argument('--experiment-id', default=None, type=str)
    argparser.add_argument('--project-id', default="gat", type=str)
    argparser.add_argument('--graph-reverse-edges', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--graph-self-loop', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--num-epochs', default=500, type=int)
    argparser.add_argument('--lr', default=0.001, type=float)
    argparser.add_argument('--node-hidden-feats', default=128, type=int)
    argparser.add_argument('--edge-hidden-feats', default=0, type=int)
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
        utils.download_dataset(args.dataset)

    if args.experiment_id is not None:
        if os.getenv('SIGOPT_API_TOKEN') is None:
            raise ValueError(
                'SigOpt API token is not provided. Please provide it by '
                '--sigopt-api-token argument or set '
                'SIGOPT_API_TOKEN environment variable.'
            )
        sigopt.set_project(args.project_id)
        experiment = sigopt.get_experiment(args.experiment_id)
        while not experiment.is_finished():
            with experiment.create_run() as sigopt_context:
                try:
                    run(args, sigopt_context=sigopt_context)
                except:
                    sigopt_context.log_failure()
    else:
        run(args)