import argparse
import os
from timeit import default_timer
from typing import Callable, Union

import dgl
import sigopt
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator

import utils
from model import EntityClassify, RelGraphEmbedding


def train(
    embedding_layer: nn.Module,
    model: nn.Module,
    device: Union[str, torch.device],
    embedding_optimizer: torch.optim.Optimizer,
    model_optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    evaluator: Evaluator,
    dataloader: dgl.dataloading.NodeDataLoader,
    labels: torch.Tensor,
    predict_category: str,
) -> tuple[float]:
    model.train()

    total_loss = 0
    total_score = 0

    start = default_timer()

    for step, (in_nodes, out_nodes, blocks) in enumerate(dataloader):
        embedding_optimizer.zero_grad()
        model_optimizer.zero_grad()

        out_nodes = out_nodes[predict_category]
        blocks = [block.int().to(device) for block in blocks]

        batch_labels = labels[out_nodes]

        embedding = embedding_layer(in_nodes)
        logits = model(blocks, embedding)[predict_category]

        loss = loss_function(logits, batch_labels)
        score = utils.get_evaluation_score(evaluator, logits, batch_labels)

        loss.backward()
        model_optimizer.step()
        embedding_optimizer.step()

        total_loss += loss.item()
        total_score += score

    stop = default_timer()
    time = stop - start

    total_loss /= step + 1
    total_score /= step + 1

    return time, total_loss, total_score


def validate(
    embedding_layer: nn.Module,
    model: nn.Module,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    evaluator: Evaluator,
    hg: dgl.DGLHeteroGraph,
    labels: torch.Tensor,
    predict_category: str,
    mask: torch.Tensor,
) -> tuple[float]:
    embedding_layer.eval()
    model.eval()

    start = default_timer()

    valid_labels = labels[mask]

    with torch.no_grad():
        embedding = embedding_layer()
        logits = model(hg, embedding)[predict_category][mask]

        loss = loss_function(logits, valid_labels)
        score = utils.get_evaluation_score(evaluator, logits, valid_labels)

    stop = default_timer()
    time = stop - start

    loss = loss.item()

    return time, loss, score


def run(
    args: argparse.ArgumentParser,
    sigopt_context: sigopt.run_context = None,
) -> None:
    torch.manual_seed(args.seed)

    dataset, evaluator, hg, train_idx, valid_idx, test_idx = utils.process_dataset(
        args.dataset,
        root=args.dataset_root,
    )
    predict_category = dataset.predict_category
    labels = hg.nodes[predict_category].data['labels']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if sigopt_context is not None:
        embedding_lr = sigopt_context.params.embedding_lr
        model_lr = sigopt_context.params.model_lr
        hidden_feats = sigopt_context.params.hidden_feats
        num_bases = sigopt_context.params.num_bases
        num_layers = sigopt_context.params.num_layers
        norm = sigopt_context.params.norm
        batch_norm = bool(sigopt_context.params.batch_norm)
        activation = sigopt_context.params.activation
        input_dropout = sigopt_context.params.input_dropout
        dropout = sigopt_context.params.dropout
        self_loop = bool(sigopt_context.params.self_loop)
        batch_size = sigopt_context.params.batch_size
        fanouts = utils.set_fanouts(
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
        embedding_lr = args.embedding_lr
        model_lr = args.model_lr
        hidden_feats = args.hidden_feats if len(
            args.hidden_feats) > 1 else args.hidden_feats[0]
        num_bases = args.num_bases
        num_layers = args.num_layers
        norm = args.norm
        batch_norm = args.batch_norm
        activation = args.activation
        input_dropout = args.input_dropout
        dropout = args.dropout
        self_loop = args.self_loop
        batch_size = args.batch_size
        fanouts = args.fanouts

    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts)
    train_dataloader = dgl.dataloading.NodeDataLoader(
        hg,
        {predict_category: train_idx},
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
    )

    in_feats = hg.nodes[predict_category].data['feat'].shape[-1]
    out_feats = dataset.num_classes
    num_nodes = {}
    node_feats = {}

    for ntype in hg.ntypes:
        num_nodes[ntype] = hg.num_nodes(ntype)
        node_feats[ntype] = hg.nodes[ntype].data.get('feat')

    activations = {'leaky_relu': F.leaky_relu, 'relu': F.relu}

    embedding_layer = RelGraphEmbedding(
        hg,
        in_feats,
        num_nodes,
        node_feats,
    )

    model = EntityClassify(
        hg,
        in_feats,
        hidden_feats,
        out_feats,
        num_bases,
        num_layers,
        norm=norm,
        batch_norm=batch_norm,
        input_dropout=input_dropout,
        dropout=dropout,
        activation=activations[activation],
        self_loop=self_loop,
    )

    loss_function = nn.CrossEntropyLoss().to(device)
    embedding_optimizer = torch.optim.SparseAdam(list(
        embedding_layer.node_embeddings.parameters()),
        lr=embedding_lr
    )
    model_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_lr
    )

    checkpoint = utils.Callback(
        args.early_stopping_patience,
        args.early_stopping_monitor,
        timeout=18_000,
    )

    for epoch in range(args.num_epochs):
        train_time, train_loss, train_score = train(
            embedding_layer,
            model,
            device,
            embedding_optimizer,
            model_optimizer,
            loss_function,
            evaluator,
            train_dataloader,
            labels,
            predict_category,
        )
        valid_time, valid_loss, valid_score = validate(
            embedding_layer,
            model,
            loss_function,
            evaluator,
            hg,
            labels,
            predict_category,
            valid_idx,
        )

        checkpoint.create(
            epoch,
            train_time,
            valid_time,
            train_loss,
            valid_loss,
            train_score,
            valid_score,
            {'embedding_layer': embedding_layer, 'model': model},
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
        embedding_layer.load_state_dict(
            checkpoint.best_epoch_model_parameters['embedding_layer'])
        model.load_state_dict(checkpoint.best_epoch_model_parameters['model'])

        test_time, test_loss, test_score = validate(
            embedding_layer,
            model,
            loss_function,
            evaluator,
            hg,
            labels,
            predict_category,
            test_idx,
        )

        print(
            f'Test Loss: {test_loss:.2f} '
            f'Test Score: {test_score:.4f} '
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


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GraphSAGE NS Optimization')

    argparser.add_argument('--dataset', default='ogbn-mag', type=str,
                           choices=['ogbn-mag'])
    argparser.add_argument('--dataset-root', default='dataset', type=str)
    argparser.add_argument('--download-dataset', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--sigopt-api-token', default=None, type=str)
    argparser.add_argument('--experiment-id', default=None, type=str)
    argparser.add_argument('--project-id', default="rgcn", type=str)
    argparser.add_argument('--num-epochs', default=500, type=int)
    argparser.add_argument('--embedding-lr', default=0.01, type=float)
    argparser.add_argument('--model-lr', default=0.01, type=float)
    argparser.add_argument('--hidden-feats', default=64, nargs='+', type=int)
    argparser.add_argument('--num-bases', default=2, type=int)
    argparser.add_argument('--num-layers', default=2, type=int)
    argparser.add_argument('--norm', default='right',
                           type=str, choices=['both', 'none', 'right'])
    argparser.add_argument('--batch-norm', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--input-dropout', default=0.1, type=float)
    argparser.add_argument('--dropout', default=0.5, type=float)
    argparser.add_argument('--activation', default='relu',
                           type=str, choices=['leaky_relu', 'relu'])
    argparser.add_argument('--self-loop', default=True,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--batch-size', default=1024, type=int)
    argparser.add_argument('--fanouts', default=[5, 10, 15],
                           nargs='+', type=str)
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
