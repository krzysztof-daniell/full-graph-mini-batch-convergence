import argparse
import os
from timeit import default_timer
from typing import Callable

import dgl
import numpy as np
import sigopt
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator

import utils
from model import GraphSAGE


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
        hidden_feats = sigopt_context.params.hidden_feats
        num_layers = sigopt_context.params.num_layers
        aggregator_type = sigopt_context.params.aggregator_type
        batch_norm = sigopt_context.params.batch_norm
        activation = sigopt_context.params.activation
        input_dropout = sigopt_context.params.input_dropout
        dropout = sigopt_context.params.dropout

        print(f'{sigopt_context.params = }')
    else:
        lr = args.lr
        hidden_feats = args.hidden_feats if len(
            args.hidden_feats) > 1 else args.hidden_feats[0]
        num_layers = args.num_layers
        aggregator_type = args.aggregator_type
        batch_norm = args.batch_norm
        activation = args.activation
        input_dropout = args.input_dropout
        dropout = args.dropout

    in_feats = g.ndata['feat'].shape[-1]
    out_feats = dataset.num_classes

    activations = {'leaky_relu': F.leaky_relu, 'relu': F.relu}

    model = GraphSAGE(
        in_feats,
        hidden_feats,
        out_feats,
        num_layers,
        aggregator_type=aggregator_type,
        batch_norm=batch_norm,
        input_dropout=input_dropout,
        dropout=dropout,
        activation=activations[activation],
    ).to(device)

    if args.dataset == 'ogbn-proteins':
        loss_function = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_function = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint = utils.Callback(
        args.early_stopping_patience,
        args.early_stopping_monitor,
        timeout=25_200,
        log_checkpoint_every=np.ceil(args.num_epochs / 200),
    )

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
            sigopt_context=sigopt_context,
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
        elif checkpoint.timeout:
            print('!! Timeout !!')

            break

    if args.test_validation:
        model.load_state_dict(checkpoint.best_epoch_model_parameters)

        test_time, test_loss, test_score = validate(
            model, loss_function, evaluator, g, test_idx)

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
            'avg train epoch time': checkpoint.avg_train_time,
            'avg valid epoch time': checkpoint.avg_valid_time,
            'experiment time': checkpoint.experiment_time,
        }

        if args.test_validation:
            metrics['best epoch - test loss'] = test_loss
            metrics['best epoch - test score'] = test_score
            metrics['test epoch time'] = test_time

        utils.log_metrics_to_sigopt(sigopt_context, checkpoint, **metrics)

    if args.save_checkpoints_to_csv:
        if sigopt_context is not None:
            path = f'{args.checkpoints_path}_{sigopt_context.id}.csv'
        elif args.checkpoints_path is None:
            path = f'full_graph_{args.dataset.replace("-", "_")}.csv'
        else:
            path = args.checkpoints_path

        utils.save_checkpoints_to_csv(checkpoint, path)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GraphSAGE Optimization')

    argparser.add_argument('--dataset', default='ogbn-products', type=str,
                           choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins'])
    argparser.add_argument('--dataset-root', default='dataset', type=str)
    argparser.add_argument('--download-dataset', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--sigopt-api-token', default=None, type=str)
    argparser.add_argument('--experiment-id', default=None, type=str)
    argparser.add_argument('--project-id', default="graphsage", type=str)
    argparser.add_argument('--graph-reverse-edges', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--graph-self-loop', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--num-epochs', default=1000, type=int)
    argparser.add_argument('--lr', default=0.01, type=float)
    argparser.add_argument('--hidden-feats', default=[256],
                           nargs='+', type=int)
    argparser.add_argument('--num-layers', default=3, type=int)
    argparser.add_argument('--aggregator-type', default='mean',
                           type=str, choices=['gcn', 'mean'])
    argparser.add_argument('--batch-norm', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--input-dropout', default=0.1, type=float)
    argparser.add_argument('--dropout', default=0.5, type=float)
    argparser.add_argument('--activation', default='relu',
                           type=str, choices=['leaky_relu', 'relu'])
    argparser.add_argument('--early-stopping-patience', default=10, type=int)
    argparser.add_argument('--early-stopping-monitor',
                           default='loss', type=str)
    argparser.add_argument('--test-validation', default=True,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--seed', default=13, type=int)
    argparser.add_argument('--save-checkpoints-to-csv', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--checkpoints-path',
                           default='checkpoints', type=str)

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
                except Exception as e:
                    print(f"Exception occurred: '{e}'")
                    sigopt_context.log_failure()
                    import sys
                    sys.exit()
    else:
        run(args)
