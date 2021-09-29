import os
import shutil
import sys
from copy import deepcopy
from typing import Union

import dgl
import dgl.function as fn
import matplotlib.pyplot as plt
import numpy as np
import psutil
import sigopt
import torch
import torch.nn as nn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


class Callback:
    def __init__(
        self,
        patience: int,
        monitor: str,
        timeout: float = None,
    ) -> None:
        self._patience = patience
        self._monitor = monitor
        self._timeout = timeout
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
    def best_epoch_model_parameters(
        self,
    ) -> Union[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        return self._model_parameters

    @property
    def should_stop(self) -> bool:
        return self._lookback >= self._patience

    @property
    def timeout(self) -> bool:
        experiment_time = sum(self._train_times) + sum(self._valid_times)

        return experiment_time >= self._timeout

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
        sigopt_context=None,
    ) -> None:
        self._train_times.append(train_time)
        self._valid_times.append(valid_time)
        self._train_losses.append(train_loss)
        self._valid_losses.append(valid_loss)
        self._train_accuracies.append(train_accuracy)
        self._valid_accuracies.append(valid_accuracy)

        if sigopt_context is not None and epoch % 5 == 0:
            sigopt_context.log_checkpoint({
                'train loss': train_loss,
                'valid loss': valid_loss,
                'train accuracy': train_accuracy,
                'valid accuracy': valid_accuracy,
            })

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
                    self._model_parameters[name] = deepcopy(
                        current_model.state_dict())
            else:
                self._model_parameters = deepcopy(model.state_dict())

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

    h_accuracy, l_accuracy = ax_accuracy.get_legend_handles_labels()
    h_loss, l_loss = ax_loss.get_legend_handles_labels()

    ax_accuracy.legend(
        handles=h_accuracy + h_loss,
        labels=l_accuracy + l_loss,
        bbox_to_anchor=(0.5, 1.01),
        loc='lower center',
        ncol=2,
    )

    plt.show()

    return fig


def log_metrics_to_sigopt(
    sigopt_context: sigopt.run_context,
    **metrics,
) -> None:
    for name, value in metrics.items():
        sigopt_context.log_metric(name=name, value=value)


def download_dataset(dataset: str) -> None:
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
    def num_labels(self) -> int:
        return self._num_labels

    @property
    def num_classes(self) -> int:
        return self._num_labels

    @property
    def predict_category(self) -> str:
        return self._predict_category

    def __getitem__(self, idx: int) -> Union[dgl.DGLGraph, dgl.DGLHeteroGraph]:
        return self._g


def load_ogbn_homogeneous(name: str, root: str = None) -> OGBDataset:
    dataset = DglNodePropPredDataset(name, root=root)

    split_idx = dataset.get_idx_split()

    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    g, labels = dataset[0]

    labels = labels.squeeze()

    if name == 'ogbn-proteins':
        g.update_all(fn.copy_e('feat', 'feat_copy'),
                     fn.sum('feat_copy', 'feat'))

        g.ndata['label'] = labels.float()
        num_labels = labels.shape[-1]
    else:
        g.ndata['label'] = labels
        num_labels = len(torch.unique(
            g.ndata['label'][torch.logical_not(torch.isnan(labels))]))

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


def process_dataset(
    name: str,
    root: str = None,
    reverse_edges: bool = False,
    self_loop: bool = False,
) -> tuple[Union[dgl.DGLGraph, dgl.DGLHeteroGraph], torch.Tensor]:
    if root is None:
        root = 'datasets'

    ogbn_homogeneous = ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins']

    if name in ogbn_homogeneous:
        dataset = load_ogbn_homogeneous(name, root=root)
    elif name == 'ogbn-mag':
        dataset = load_ogbn_mag(root=root)

    evaluator = Evaluator(name)
    g = dataset[0]

    if reverse_edges:
        src, dst = g.all_edges()

        g.add_edges(dst, src)

    if self_loop:
        g = g.remove_self_loop().add_self_loop()

    if name in ogbn_homogeneous:
        train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
        valid_idx = torch.nonzero(g.ndata['valid_mask'], as_tuple=True)[0]
        test_idx = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
    elif name == 'ogbn-mag':
        predict_category = dataset.predict_category

        train_idx = torch.nonzero(
            g.nodes[predict_category].data['train_mask'], as_tuple=True)[0]
        valid_idx = torch.nonzero(
            g.nodes[predict_category].data['valid_mask'], as_tuple=True)[0]
        test_idx = torch.nonzero(
            g.nodes[predict_category].data['test_mask'], as_tuple=True)[0]

    return dataset, evaluator, g, train_idx, valid_idx, test_idx


def set_sigopt_fanouts(fanouts: str) -> list[int]:
    default_fanouts = [int(i) for i in fanouts.split(',')]
    sigopt_fanouts = []

    for i in range(sigopt.get_parameter('num_layers', default=len(default_fanouts))):
        if i < len(default_fanouts):
            fanout = sigopt.get_parameter(
                f'layer_{i + 1}_fanout', default=default_fanouts[i])
        else:
            fanout = sigopt.get_parameter(f'layer_{i + 1}_fanout')

        sigopt_fanouts.append(fanout)

    return sigopt_fanouts


def log_system_info() -> None:
    # https://psutil.readthedocs.io/en/latest/#processes
    process = psutil.Process()
    virtual_memory = psutil.virtual_memory()
    sigopt.log_metadata("Python version", sys.version.split()[0])
    sigopt.log_metadata("Operating System", sys.platform)
    sigopt.log_metadata("psutil.Process().num_threads", process.num_threads())
    # run.log_metadata("Process CPU Percent", process.cpu_percent())
    sigopt.log_metadata("psutil.virtual_memory().total",
                        psutil._common.bytes2human(virtual_memory.total))
    sigopt.log_metadata("psutil.virtual_memory().available",
                        psutil._common.bytes2human(virtual_memory.available))
    # run.log_metadata("Virtual Memory Percent", virtual_memory.percent)


def get_evaluation_score(
    evaluator: Evaluator,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    if labels.dim() > 1:
        y_pred = logits
        y_true = labels
    else:
        y_pred = logits.argmax(dim=-1, keepdim=True)
        y_true = labels.unsqueeze(dim=-1)

    _, score = evaluator.eval({
        'y_pred': y_pred,
        'y_true': y_true,
    }).popitem()

    return score


def is_experiment_finished(experiment) -> bool:
    observation_count = experiment.fetch().progress.observation_count
    observation_budget = experiment.fetch().observation_budget

    return observation_count <= observation_budget
