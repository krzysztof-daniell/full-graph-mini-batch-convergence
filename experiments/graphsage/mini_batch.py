import argparse
import os
from timeit import default_timer
from typing import Callable, Union

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
    device: Union[str, torch.device],
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    evaluator: Evaluator,
    dataloader: dgl.dataloading.NodeDataLoader,
) -> tuple[float]:
    model.train()

    total_loss = 0
    total_score = 0

    start = default_timer()

    for step, (_, _, blocks) in enumerate(dataloader):
        optimizer.zero_grad()

        blocks = [block.int().to(device) for block in blocks]

        inputs = blocks[0].srcdata['feat']
        labels = blocks[-1].dstdata['label']

        logits = model(blocks, inputs)

        loss = loss_function(logits, labels)
        score = utils.get_evaluation_score(evaluator, logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_score += score

    stop = default_timer()
    time = stop - start

    total_loss /= step + 1
    total_score /= step + 1

    return time, total_loss, total_score


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


def set_fanouts(
    num_layers: int,
    batch_size: int,
    max_num_batch_nodes: int,
    fanout_slope: float,
    max_fanout: int = 40,
) -> list[int]:
    result_fanouts = None

    for base_fanout in range(max_fanout + 1):
        fanouts = []

        for n in range(num_layers):
            fanout = int((fanout_slope ** n) * base_fanout)

            if fanout < 1:
                fanout = 1

            if fanout > max_fanout:
                break

            fanouts.append(fanout)

        if len(fanouts) == num_layers:
            result = batch_size

            for fanout in reversed(fanouts):
                result += result * fanout

            if result <= max_num_batch_nodes:
                result_fanouts = fanouts

    if result_fanouts is None:
        result_fanouts = [1 for _ in range(num_layers)]

    return result_fanouts


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
        sigopt_context.params.setdefaults({
            'lr': args.lr,
            'num_layers': args.num_layers,
            'aggregator_type': args.aggregator_type,
            'batch_norm': int(args.batch_norm),
            'activation': args.activation,
            'input_dropout': args.input_dropout,
            'dropout': args.dropout,
            'batch_size': args.batch_size,
        })

        # for i, fanout in enumerate(args.fanouts):
        #     sigopt.params.setdefault(f'fanout_layer_{i + 1}', fanout)

        for i in range(args.num_layers - 1):
            if isinstance(args.hidden_feats, int):
                hidden_feats = args.hidden_feats
            else:
                hidden_feats = args.hidden_feats[i]

            sigopt_context.params.setdefault(
                f'hidden_feats_layer_{i + 1}', hidden_feats)

        lr = sigopt_context.params['lr']
        num_layers = int(sigopt_context.params['num_layers'])
        hidden_feats = [sigopt_context.params[f'hidden_feats_layer_{i + 1}']
                        for i in range(num_layers - 1)]
        aggregator_type = sigopt_context.params['aggregator_type']
        batch_norm = sigopt_context.params['batch_norm']
        activation = sigopt_context.params['activation']
        input_dropout = sigopt_context.params['input_dropout']
        dropout = sigopt_context.params['dropout']
        batch_size = sigopt_context.params['batch_size']
        # fanouts = [sigopt_context.params[f'fanout_layer_{i + 1}']
        #            for i in range(num_layers)]

        fanouts = set_fanouts(
            num_layers,
            batch_size,
            sigopt_context.params['max_batch_num_nodes'],
            sigopt_context.params['fanout_slope'],
        )

        sigopt_context.log_metadata(
            'fanouts', ','.join([str(i) for i in fanouts]))

        print(f'{sigopt_context.params = }')
        print(f'{fanouts = }')
    else:
        lr = args.lr
        num_layers = args.num_layers
        hidden_feats = args.hidden_feats
        aggregator_type = args.aggregator_type
        batch_norm = args.batch_norm
        activation = args.activation
        input_dropout = args.input_dropout
        dropout = args.dropout
        batch_size = args.batch_size
        fanouts = args.fanouts

    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts)
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_idx,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

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

    checkpoint = utils.Callback(args.early_stopping_patience,
                                args.early_stopping_monitor)

    for epoch in range(args.num_epochs):
        train_time, train_loss, train_score = train(
            model,
            device,
            optimizer,
            loss_function,
            evaluator,
            train_dataloader,
        )
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
            'avg train epoch time': np.mean(checkpoint.train_times),
            'avg valid epoch time': np.mean(checkpoint.valid_times),
        }

        if args.test_validation:
            metrics['best epoch - test loss'] = test_loss
            metrics['best epoch - test score'] = test_score
            metrics['test epoch time'] = test_time

        utils.log_metrics_to_sigopt(sigopt_context, **metrics)

        sigopt_context.end()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GraphSAGE NS Optimization')

    argparser.add_argument('--dataset', default='ogbn-products', type=str,
                           choices=['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins'])
    argparser.add_argument('--dataset-root', default='dataset', type=str)
    argparser.add_argument('--download-dataset', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--experiment-id', default=None, type=str)
    argparser.add_argument('--project-id', default="graphsage", type=str)
    argparser.add_argument('--graph-reverse-edges', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--graph-self-loop', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--num-epochs', default=500, type=int)
    argparser.add_argument('--lr', default=0.003, type=float)
    argparser.add_argument('--hidden-feats', default=256, nargs='+', type=int)
    argparser.add_argument('--num-layers', default=3, type=int)
    argparser.add_argument('--aggregator-type', default='mean',
                           type=str, choices=['gcn', 'mean'])
    argparser.add_argument('--batch-norm', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--input-dropout', default=0.1, type=float)
    argparser.add_argument('--dropout', default=0.5, type=float)
    argparser.add_argument('--activation', default='relu',
                           type=str, choices=['leaky_relu', 'relu'])
    argparser.add_argument('--batch-size', default=1000, type=int)
    argparser.add_argument('--fanouts', default=[5, 10, 15],
                           nargs='+', type=str)
    argparser.add_argument('--num-workers', default=4, type=int)
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
                run(args, sigopt_context=sigopt_context)
    else:
        run(args)
