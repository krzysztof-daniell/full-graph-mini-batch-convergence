import argparse
import os
import shutil
from typing import Union

import dgl
import matplotlib.pyplot as plt
import numpy as np
import sigopt
import torch
import torch.nn as nn
from ogb.nodeproppred import DglNodePropPredDataset


class Callback:
    def __init__(
        self,
        patience: int,
        monitor: str,
    ) -> None:
        self._patience = patience
        self._monitor = monitor
        self._lookback = 0
        self._best_epoch = None
        self._train_times = []
        self._valid_times = []
        self._train_losses = []
        self._valid_losses = []
        self._train_accuracies = []
        self._valid_accuracies = []
        self._model_parameters = {}

    @property
    def best_epoch(self) -> int:
        return self._best_epoch + 1

    @property
    def train_times(self) -> list[float]:
        return self._train_times

    @property
    def valid_times(self) -> list[float]:
        return self._valid_times

    @property
    def train_losses(self) -> list[float]:
        return self._train_losses

    @property
    def valid_losses(self) -> list[float]:
        return self._valid_losses

    @property
    def train_accuracies(self) -> list[float]:
        return self._train_accuracies

    @property
    def valid_accuracies(self) -> list[float]:
        return self._valid_accuracies

    @property
    def best_epoch_training_time(self) -> float:
        return sum(self._train_times[:self._best_epoch])

    @property
    def best_epoch_train_loss(self) -> float:
        return self._train_losses[self._best_epoch]

    @property
    def best_epoch_valid_loss(self) -> float:
        return self._valid_losses[self._best_epoch]

    @property
    def best_epoch_train_accuracy(self) -> float:
        return self._train_accuracies[self._best_epoch]

    @property
    def best_epoch_valid_accuracy(self) -> float:
        return self._valid_accuracies[self._best_epoch]

    @property
    def best_epoch_model_parameters(self) -> Union[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        return self._model_parameters

    @property
    def should_stop(self) -> bool:
        return self._lookback >= self._patience

    def create(
        self,
        epoch: int,
        train_time: float,
        valid_time: float,
        train_loss: float,
        valid_loss: float,
        train_accuracy: float,
        valid_accuracy: float,
        model: Union[nn.Module, dict[str, nn.Module]],
    ):
        self._train_times.append(train_time)
        self._valid_times.append(valid_time)
        self._train_losses.append(train_loss)
        self._valid_losses.append(valid_loss)
        self._train_accuracies.append(train_accuracy)
        self._valid_accuracies.append(valid_accuracy)

        best_epoch = False

        if self._best_epoch is None:
            best_epoch = True
        elif self._monitor == 'loss':
            if valid_loss < self._valid_losses[self._best_epoch]:
                best_epoch = True
        elif self._monitor == 'accuracy':
            if valid_accuracy > self._valid_accuracies[self._best_epoch]:
                best_epoch = True

        if best_epoch:
            self._best_epoch = epoch

            if isinstance(model, dict):
                for name, current_model in model.items():
                    self._model_parameters[name] = current_model.state_dict()
            else:
                self._model_parameters = model.state_dict()

            self._lookback = 0
        else:
            self._lookback += 1


def get_metrics_plot(
    train_accuracies: list[float],
    valid_accuracies: list[float],
    train_losses: list[float],
    valid_losses: list[float],
) -> plt.Figure:
    fig, ax_accuracy = plt.subplots(1, 1)

    ax_accuracy.plot(np.array(train_accuracies), 'b', label='Train Accuracy')
    ax_accuracy.plot(np.array(valid_accuracies), 'r',
                     label='Validation Accuracy')
    ax_accuracy.set_xlabel('Epochs', color='black')
    ax_accuracy.set_ylabel('Accuracy', color='black')

    ax_loss = ax_accuracy.twinx()
    ax_loss.plot(np.array(train_losses), 'b--', label='Train Loss')
    ax_loss.plot(np.array(valid_losses), 'r--', label='Validation Loss')
    ax_loss.set_ylabel('Loss', color='black')
    ax_loss.legend(loc='center left')

    plt.show()

    return fig


def log_metrics_to_sigopt(
    checkpoint: Callback,
    model_name: str,
    dataset: str,
    test_loss: float,
    test_accuracy: float,
    test_time: float,
) -> None:
    sigopt.log_model(model_name)
    sigopt.log_dataset(dataset)
    sigopt.log_metric('best epoch', checkpoint.best_epoch)
    sigopt.log_metric('best epoch - train loss',
                      checkpoint.best_epoch_train_loss)
    sigopt.log_metric('best epoch - train accuracy',
                      checkpoint.best_epoch_train_accuracy)
    sigopt.log_metric('best epoch - valid loss',
                      checkpoint.best_epoch_valid_loss)
    sigopt.log_metric('best epoch - valid accuracy',
                      checkpoint.best_epoch_valid_accuracy)
    sigopt.log_metric('best epoch - test loss', test_loss)
    sigopt.log_metric('best epoch - test accuracy', test_accuracy)
    sigopt.log_metric('best epoch - training time',
                      checkpoint.best_epoch_training_time)
    sigopt.log_metric('avg train epoch time', np.mean(checkpoint.train_times))
    sigopt.log_metric('avg valid epoch time', np.mean(checkpoint.valid_times))
    sigopt.log_metric('test epoch time', test_time)

    metrics_plot = get_metrics_plot(
        checkpoint.train_accuracies,
        checkpoint.valid_accuracies,
        checkpoint.train_losses,
        checkpoint.valid_losses,
    )

    sigopt.log_image(metrics_plot, name='convergence plot')


def download_dataset(dataset: str):
    if dataset == 'ogbn-products':
        command = 'aws s3 cp s3://ogb-products ./dataset --recursive'
        os.system(command)
        shutil.move('./dataset/ogbn_products', './dataset/ogbn_products_dgl')


class OGBDataset:
    def __init__(
        self,
        g: dgl.DGLGraph,
        num_labels: int,
        predict_category: str = None,
    ) -> None:
        self._g = g
        self._num_labels = num_labels
        self._predict_category = predict_category

    @property
    def num_labels(self):
        return self._num_labels

    @property
    def num_classes(self):
        return self._num_labels

    @property
    def predict_category(self):
        return self._predict_category

    def __getitem__(self, idx: int):
        return self._g


def load_ogbn_homogeneous(name: str, root: str = None) -> OGBDataset:
    dataset = DglNodePropPredDataset(name, root=root)

    split_idx = dataset.get_idx_split()

    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    g, labels = dataset[0]

    labels = labels.squeeze()
    num_labels = len(torch.unique(
        labels[torch.logical_not(torch.isnan(labels))]))

    g.ndata['label'] = labels

    train_mask = torch.zeros((g.num_nodes(),), dtype=torch.bool)
    train_mask[train_idx] = True
    valid_mask = torch.zeros((g.num_nodes(),), dtype=torch.bool)
    valid_mask[valid_idx] = True
    test_mask = torch.zeros((g.num_nodes(),), dtype=torch.bool)
    test_mask[test_idx] = True

    g.ndata['train_mask'] = train_mask
    g.ndata['valid_mask'] = valid_mask
    g.ndata['test_mask'] = test_mask

    ogb_dataset = OGBDataset(g, num_labels)

    return ogb_dataset


def load_ogbn_mag(root: str = None) -> OGBDataset:
    dataset = DglNodePropPredDataset(name='ogbn-mag', root=root)

    split_idx = dataset.get_idx_split()

    train_idx = split_idx['train']['paper']
    valid_idx = split_idx['valid']['paper']
    test_idx = split_idx['test']['paper']

    hg_original, labels = dataset[0]

    labels = labels['paper'].squeeze()
    num_labels = dataset.num_classes

    subgraphs = {}

    for etype in hg_original.canonical_etypes:
        src, dst = hg_original.all_edges(etype=etype)

        subgraphs[etype] = (src, dst)
        subgraphs[(etype[2], f'rev-{etype[1]}', etype[0])] = (dst, src)

    hg = dgl.heterograph(subgraphs)

    hg.nodes['paper'].data['feat'] = hg_original.nodes['paper'].data['feat']
    hg.nodes['paper'].data['labels'] = labels

    train_mask = torch.zeros((hg.num_nodes('paper'),), dtype=torch.bool)
    train_mask[train_idx] = True
    valid_mask = torch.zeros((hg.num_nodes('paper'),), dtype=torch.bool)
    valid_mask[valid_idx] = True
    test_mask = torch.zeros((hg.num_nodes('paper'),), dtype=torch.bool)
    test_mask[test_idx] = True

    hg.nodes['paper'].data['train_mask'] = train_mask
    hg.nodes['paper'].data['valid_mask'] = valid_mask
    hg.nodes['paper'].data['test_mask'] = test_mask

    ogb_dataset = OGBDataset(hg, num_labels, 'paper')

    return ogb_dataset


def process_dataset(name: str, root: str = None):
    if root is None:
        root = 'datasets'

    ogbn_homogeneous = ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins']

    if name == 'reddit':
        dataset = dgl.data.RedditDataset(self_loop=True, raw_dir=root)
    elif name in ogbn_homogeneous:
        dataset = load_ogbn_homogeneous(name, root=root)
    elif name == 'ogbn-mag':
        dataset = load_ogbn_mag(root=root)

    return dataset
