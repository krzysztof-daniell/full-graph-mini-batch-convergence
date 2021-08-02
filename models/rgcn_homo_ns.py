from collections.abc import Callable
from timeit import default_timer
from typing import Union

import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn


class RelGraphEmbedLayer(nn.Module):
    def __init__(
        self,
        node_tids: torch.Tensor,
        num_ntypes: int,
        input_size: list[Union[int, None]],
        embedding_size: int,
    ):
        super().__init__()
        self._num_ntypes = num_ntypes
        self._embedding_size = embedding_size

        self.embeddings = nn.ParameterDict()

        for ntype in range(num_ntypes):
            if input_size[ntype] is not None:
                embedding = nn.Parameter(torch.Tensor(
                    input_size[ntype].shape[-1], embedding_size))
                nn.init.xavier_uniform_(embedding)

                self.embeddings[f'{ntype}'] = embedding

        self.node_embeddings = nn.Embedding(
            node_tids.shape[0], embedding_size, sparse=True)
        nn.init.uniform_(self.node_embeddings.weight, -1, 1)

    def forward(
        self,
        node_ids: torch.Tensor,
        node_tids: torch.Tensor,
        type_ids: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.empty(node_ids.shape[0], self._embedding_size)

        for ntype in range(self._num_ntypes):
            loc = node_tids == ntype

            if inputs[ntype] is not None:
                x[loc] = inputs[ntype][type_ids[loc]
                                       ] @ self.embeddings[f'{ntype}']
            else:
                x[loc] = self.node_embeddings(node_ids[loc])

        return x


class EntityClassify(nn.Module):
    def __init__(
        self,
        hidden_feats: int,
        out_feats: int,
        num_rels: int,
        num_layers: int,
        num_bases: int = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
        dropout: float = 0,
        self_loop: bool = False,
        layer_norm: bool = False,

    ):
        super().__init__()

        self._layers = nn.ModuleList()

        self._layers.append(dglnn.RelGraphConv(
            hidden_feats,
            hidden_feats,
            num_rels,
            'basis',
            num_bases,
            activation=activation,
            self_loop=self_loop,
            low_mem=True,
            dropout=dropout,
            layer_norm=layer_norm,
        ))

        for _ in range(1, num_layers - 1):
            self._layers.append(dglnn.RelGraphConv(
                hidden_feats,
                hidden_feats,
                num_rels,
                'basis',
                num_bases,
                activation=activation,
                self_loop=self_loop,
                low_mem=True,
                dropout=dropout,
                layer_norm=layer_norm,
            ))

        self._layers.append(dglnn.RelGraphConv(
            hidden_feats,
            out_feats,
            num_rels,
            'basis',
            num_bases,
            activation=None,
            self_loop=self_loop,
            low_mem=True,
            dropout=dropout,
            layer_norm=layer_norm,
        ))

    def forward(
        self,
        blocks: dgl.DGLGraph,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = inputs

        for layer, block in zip(self._layers, blocks):
            x = layer(block, x, block.edata['etype'], block.edata['norm'])

        return x

    def inference(
        self,
        g: dgl.DGLGraph,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = inputs

        for layer in self._layers:
            x = layer(g, x, g.edata['etype'], g.edata['norm'])
            # x = layer(g, x, g.edata[dgl.ETYPE], g.edata['norm'])  # TODO: try this version

        return x


def train(
    embedding_layer: nn.Module,
    model: nn.Module,
    embedding_optimizer: torch.optim.Optimizer,
    model_optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    node_feats: torch.Tensor,
    labels: torch.Tensor,
    dataloader: dgl.dataloading.NodeDataLoader,
) -> tuple[float]:
    embedding_layer.train()
    model.train()

    total_loss = 0
    total_accuracy = 0

    start = default_timer()

    for step, (in_nodes, _, blocks) in enumerate(dataloader):
        embedding_optimizer.zero_grad()
        model_optimizer.zero_grad()

        batch_labels = labels[blocks[-1].dstdata['type_id']]

        inputs = embedding_layer(
            in_nodes,
            blocks[0].srcdata['ntype'],
            blocks[0].srcdata['type_id'],
            node_feats,
        )
        logits = model(blocks, inputs)
        loss = loss_function(logits, batch_labels)

        loss.backward()
        model_optimizer.step()
        embedding_optimizer.step()

        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == batch_labels)
        accuracy = correct.item() / len(batch_labels)

        total_loss += loss.item()
        total_accuracy += accuracy

    stop = default_timer()
    time = stop - start

    total_loss /= step + 1
    total_accuracy /= step + 1

    return time, total_loss, total_accuracy


def validate_full(
    embedding_layer: nn.Module,
    model: nn.Module,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    node_feats: torch.Tensor,
    labels: torch.Tensor,
    g: dgl.DGLGraph,
    mask: torch.Tensor,
) -> tuple[float]:
    embedding_layer.eval()
    model.eval()

    inference_labels = labels[mask]

    start = default_timer()

    with torch.no_grad():
        inputs = embedding_layer(
            g.srcdata[dgl.NID],
            # g.srcdata[dgl.NTYPE],
            g.srcdata['ntype'],
            g.srcdata['type_id'],
            node_feats,
        )
        logits = model.inference(g, inputs)[mask]
        loss = loss_function(logits, inference_labels)

    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == inference_labels)
    accuracy = correct.item() / len(inference_labels)

    stop = default_timer()
    time = stop - start

    return time, loss, accuracy


def validate(
    embedding_layer: nn.Module,
    model: nn.Module,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    node_feats: torch.Tensor,
    labels: torch.Tensor,
    dataloader: dgl.dataloading.NodeDataLoader,
) -> tuple[float]:
    embedding_layer.eval()
    model.eval()

    total_loss = 0
    total_accuracy = 0

    start = default_timer()

    with torch.no_grad():
        for step, (_, _, blocks) in enumerate(dataloader):
            batch_labels = labels[blocks[-1].dstdata['type_id']]

            inputs = embedding_layer(
                blocks[0].srcdata[dgl.NID],
                blocks[0].srcdata[dgl.NTYPE],
                blocks[0].srcdata['type_id'],
                node_feats,
            )
            logits = model(blocks, inputs)
            loss = loss_function(logits, batch_labels)

            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == batch_labels)
            accuracy = correct.item() / len(batch_labels)

            total_loss += loss.item()
            total_accuracy += accuracy

    stop = default_timer()
    time = stop - start

    total_loss /= step + 1
    total_accuracy /= step + 1

    return time, loss, accuracy
