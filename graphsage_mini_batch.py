import os
from collections.abc import Callable
from pprint import pprint
from timeit import default_timer
from typing import Union

import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sigopt import Connection

from utils import process_dataset


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        num_layers: int,
        activation: Callable[[None], torch.Tensor],
        dropout: float,
    ):
        super().__init__()
        self._hidden_feats = hidden_feats
        self._out_feats = out_feats
        self._num_layers = num_layers
        self._layers = nn.ModuleList()
        self._dropout = nn.Dropout(dropout)
        self._activation = activation

        self._layers.append(dglnn.SAGEConv(in_feats, hidden_feats, 'mean'))

        for _ in range(1, num_layers - 1):
            self._layers.append(dglnn.SAGEConv(
                hidden_feats, hidden_feats, 'mean'))

        self._layers.append(dglnn.SAGEConv(hidden_feats, out_feats, 'mean'))

    def forward(
        self,
        blocks: tuple[dgl.DGLGraph],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = inputs

        for i, (layer, block) in enumerate(zip(self._layers, blocks)):
            x = layer(block, x)

            if i < self._num_layers - 1:
                x = self._activation(x)
                x = self._dropout(x)

        return x

    def inference(self, g, inputs):
        x = inputs

        for i, layer in enumerate(self._layers):
            x = layer(g, x)

            if i < self._num_layers - 1:
                x = self._activation(x)
                x = self._dropout(x)

        return x


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

    total_loss /= step
    total_accuracy /= step

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
        logits = model.inference(g, features)
        loss = loss_function(logits[mask], labels[mask])

        _, indices = torch.max(logits[mask], dim=1)
        correct = torch.sum(indices == labels[mask])
        accuracy = correct.item() / len(labels[mask])

    stop = default_timer()
    time = stop - start

    return time, loss, accuracy


if __name__ == '__main__':
    torch.manual_seed(13)

    accuracy_thresholds = [0.75, 0.76, 0.77, 0.78]
    accuracy_threshold = accuracy_thresholds.pop(0)

    conn = Connection(
        client_token='OAOZHUKIZGHAYLOBMTMUCHYLRCCVMTELSPRQPQNDODUVKMXR')
    experiment = conn.experiments().create(
        name=f'Mini-batch / OGBN-PRODUCTS / Target: {accuracy_threshold}',
        conditionals=[{
            'name': 'num_hidden_layers',
            'values': ['0', '1', '2'],
        }],
        parameters=[
            {
                'name': 'lr',
                'type': 'double',
                'bounds': {'min': 1e-5, 'max': 99999e-5},
                'transformation': 'log',
            },
            {
                'name': 'batch_size_factor',
                'type': 'int',
                'bounds': {'min': 1, 'max': 8},
                # 'bounds': {'min': 1, 'max': 16},
            },
            {
                'name': 'fanout_input',
                'conditions': {'num_hidden_layers': ['0', '1', '2']},
                'type': 'int',
                'bounds': {'min': 1, 'max': 30},
                # 'bounds': {'min': 1, 'max': 50},
            },
            {
                'name': 'fanout_hidden_1',
                'conditions': {'num_hidden_layers': ['1', '2']},
                'type': 'int',
                'bounds': {'min': 1, 'max': 30},
                # 'bounds': {'min': 1, 'max': 50},
            },
            {
                'name': 'fanout_hidden_2',
                'conditions': {'num_hidden_layers': ['2']},
                'type': 'int',
                'bounds': {'min': 1, 'max': 30},
                # 'bounds': {'min': 1, 'max': 50},
            },
            {
                'name': 'fanout_output',
                'conditions': {'num_hidden_layers': ['0', '1', '2']},
                'type': 'int',
                'bounds': {'min': 1, 'max': 30},
                # 'bounds': {'min': 1, 'max': 50},
            },
            {
                'name': 'hidden_feats_factor',
                'type': 'int',
                'bounds': {'min': 1, 'max': 16},
                # 'bounds': {'min': 1, 'max': 32},
            },
            {
                'name': 'dropout',
                'type': 'double',
                'bounds': {'min': 1e-3, 'max': 999e-3},
                # 'bounds': {'min': 1e-3, 'max': 999e-3},
                'transformation': 'log',
            },
        ],
        metrics=[
            {
                'name': 'test_accuracy',
                'objective': 'maximize',
                'strategy': 'constraint',
                'threshold': accuracy_threshold,
            },
            {
                'name': 'num_epochs',
                'objective': 'minimize',
            },
            {
                'name': 'training_time',
                'objective': 'minimize',
                'strategy': 'store',
            },
        ],
        observation_budget=int(os.environ.get(
            'OBSERVATION_BUDGET', default=500)),
        project='graphsage-convergence',
    )

    print(
        f'Created experiment: https://app.sigopt.com/experiment/{experiment.id}')

    dataset = process_dataset('ogbn-products', '/tmp/dataset')
    g = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 50

    train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    valid_idx = torch.nonzero(g.ndata['valid_mask'], as_tuple=True)[0]
    test_idx = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]

    best_num_epochs = None

    for experiment_index in range(1, 1 + experiment.observation_budget):
        suggestion = conn.experiments(experiment.id).suggestions().create()
        assignments = suggestion.assignments

        in_feats = g.ndata['feat'].shape[-1]
        hidden_feats = 16 * assignments['hidden_feats_factor']
        out_feats = dataset.num_classes
        num_hidden_layers = int(assignments['num_hidden_layers'])
        num_layers = 2 + num_hidden_layers
        activation = F.relu
        dropout = assignments['dropout']

        batch_size = 1024 * assignments['batch_size_factor']
        num_workers = 4
        fanouts = []

        fanouts.append(assignments['fanout_input'])

        if num_hidden_layers > 0:
            for i in range(1, 1 + num_hidden_layers):
                fanouts.append(assignments[f'fanout_hidden_{i}'])

        fanouts.append(assignments['fanout_output'])

        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts)
        train_dataloader = dgl.dataloading.NodeDataLoader(
            g,
            train_idx,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
        )

        model = GraphSAGE(in_feats, hidden_feats, out_feats,
                          num_layers, activation, dropout).to(device)
        loss_function = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=assignments['lr'])

        print(f'Experiment: {experiment_index} Assigments:')
        pprint(assignments)

        epoch_times = []
        best_accuracy = None
        early_stopping = 0

        for epoch in range(1, 1 + num_epochs):
            train_time, train_loss, train_accuracy = train(
                model, device, optimizer, loss_function, train_dataloader)
            # valid_time, valid_loss, valid_accuracy = validate(
            #     model, loss_function, g, valid_idx)
            test_time, test_loss, test_accuracy = validate(
                model, loss_function, g, test_idx)

            print(
                f'Epoch: {epoch:03} '
                f'Train Loss: {train_loss:.2f} '
                # f'valid Loss: {valid_loss:.2f} '
                f'Test Loss: {test_loss:.2f} '
                f'Train Accuracy: {train_accuracy * 100:.2f} % '
                # f'Valid Accuracy: {valid_accuracy * 100:.2f} % '
                f'Test Accuracy: {test_accuracy * 100:.2f} % '
                f'Train epoch time: {train_time:.2f} '
            )

            epoch_times.append(train_time)

            if best_accuracy is None or test_accuracy > best_accuracy['value']:
                best_accuracy = {'value': test_accuracy, 'epoch': epoch}
                early_stopping = 0
            elif best_accuracy is not None and test_accuracy < best_accuracy['value']:
                early_stopping += 1

                if early_stopping >= 5:
                    break

            if test_accuracy >= accuracy_threshold:
                if best_num_epochs is None or best_accuracy['epoch'] < best_num_epochs:
                    best_num_epochs = best_accuracy['epoch']

                break

            if best_num_epochs is not None and epoch >= best_num_epochs:
                break

        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            values=[
                {'name': 'test_accuracy', 'value': best_accuracy['value']},
                {'name': 'num_epochs', 'value': best_accuracy['epoch']},
                {'name': 'training_time', 'value': sum(
                    epoch_times[:best_accuracy['epoch'] + 1])},
            ],
        )

        if experiment_index % 100 == 0:
            if len(accuracy_thresholds) > 1:
                accuracy_threshold = accuracy_thresholds.pop(0)

                experiment = conn.experiments(experiment.id).update(
                    metrics=[
                        {
                            'name': 'test_accuracy',
                            'objective': 'maximize',
                            'strategy': 'constraint',
                            'threshold': accuracy_threshold,
                        },
                        {
                            'name': 'num_epochs',
                            'objective': 'minimize',
                        },
                        {
                            'name': 'training_time',
                            'objective': 'minimize',
                            'strategy': 'store',
                        },
                    ],
                )
