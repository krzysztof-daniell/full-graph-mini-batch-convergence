import os
from collections.abc import Callable
from timeit import default_timer

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
        hidden_feats: list,
        out_feats: int,
        activation: Callable[[None], torch.Tensor],
        dropout: float,
    ):
        super().__init__()
        self._hidden_feats = hidden_feats
        self._out_feats = out_feats
        self._num_layers = len(hidden_feats) 
        self._layers = nn.ModuleList()
        self._dropout = nn.Dropout(dropout)
        self._activation = activation

        self._layers.append(dglnn.SAGEConv(in_feats, hidden_feats[0], 'mean'))
        prev = hidden_feats[0]
        for i in range(1, self._num_layers - 1):
            self._layers.append(dglnn.SAGEConv(prev, hidden_feats[i], 'mean'))
            prev = hidden_feats[i]
        self._layers.append(dglnn.SAGEConv(prev, out_feats, 'mean'))

        # self._layers.append(dglnn.SAGEConv(in_feats, hidden_feats, 'mean'))

        # for _ in range(1, num_layers - 1):
        #     self._layers.append(dglnn.SAGEConv(
        #         hidden_feats, hidden_feats, 'mean'))

        # self._layers.append(dglnn.SAGEConv(hidden_feats, out_feats, 'mean'))

    def forward(
        self,
        g: dgl.DGLGraph,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = inputs

        for i, layer in enumerate(self._layers):
            x = layer(g, x)

            if i < self._num_layers - 1:
                x = self._activation(x)
                x = self._dropout(x)

        return x


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    g: dgl.DGLGraph,
    mask: torch.Tensor,
) -> tuple[float]:
    features = g.ndata['feat']
    labels = g.ndata['label']

    model.train()
    optimizer.zero_grad()

    start = default_timer()

    logits = model(g, features)
    loss = loss_function(logits[mask], labels[mask])

    loss.backward()
    optimizer.step()

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


if __name__ == '__main__':
    torch.manual_seed(13)

    # accuracy_thresholds = [0.625, 0.65, 0.675, 0.7]
    accuracy_thresholds = [0.7]

    for accuracy_threshold in accuracy_thresholds:
        conn = Connection(
            client_token='GHTSWEVLVQGIDOKKDXJWHJXMYSGYPBQDCWUXPUPADLNZSANX')
        experiment = conn.experiments().create(
            name=f'Full-graph / OGBN-PRODUCTS / Target: {accuracy_threshold}',
            parameters=[
                {
                    'name': 'lr',
                    'type': 'double',
                    'bounds': {'min': 1e-5, 'max': 9e-1},
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
            observation_budget=int(os.environ.get('OBSERVATION_BUDGET', default=10)),
            project='graphsage-convergence',
        )

        print(
            f'Created experiment: https://app.sigopt.com/experiment/{experiment.id}')

        dataset = process_dataset('ogbn-products', '/tmp/dataset')
        g = dataset[0]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        in_feats = g.ndata['feat'].shape[-1]
        hidden_feats = 16
        out_feats = dataset.num_classes
        num_layers = 2
        activation = F.relu
        dropout = 0.5

        fanouts = [10, 25]
        batch_size = 1024
        num_workers = 4

        num_epochs = 300

        train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
        valid_idx = torch.nonzero(g.ndata['valid_mask'], as_tuple=True)[0]
        test_idx = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]

        best_num_epochs = None

        for i in range(experiment.observation_budget):
            suggestion = conn.experiments(experiment.id).suggestions().create()
            assignments = suggestion.assignments

            model = GraphSAGE(in_feats, hidden_feats, out_feats,
                              num_layers, activation, dropout).to(device)
            loss_function = nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=assignments['lr'])

            print(f'Experiment: {i + 1} Learning Rate: {assignments["lr"]}')

            epoch_times = []
            best_accuracy = None
            early_stopping = 0

            for epoch in range(1, 1 + num_epochs):
                train_time, train_loss, train_accuracy = train(
                    model, optimizer, loss_function, g, train_idx)
                # valid_time, valid_loss, valid_accuracy = validate(
                #     model, device, loss_function, g, valid_idx, batch_size)
                test_time, test_loss, test_accuracy = validate(
                    model, loss_function, g, test_idx)

                print(
                    f'Epoch: {epoch + 1:03} '
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
                        epoch_times[:best_accuracy['epoch'] + 1])}
                ],
            )