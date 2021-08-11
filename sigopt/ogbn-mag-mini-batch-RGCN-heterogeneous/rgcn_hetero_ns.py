from collections.abc import Callable
from timeit import default_timer
from typing import Union

import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn

class RelGraphConvLayer(nn.Module):
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        rel_names: list[str],
        num_bases: int,
        weight: bool = True,
        bias: bool = True,
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
        dropout: float = None,
        self_loop: bool = False,
    ):
        super().__init__()
        self._rel_names = rel_names
        self._num_rels = len(rel_names)
        self._conv = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(
            in_feats, out_feats, norm='right', weight=False, bias=False) for rel in rel_names})
        self._use_weight = weight
        self._use_basis = num_bases < self._num_rels and weight
        self._use_bias = bias
        self._activation = activation
        self._dropout = nn.Dropout(dropout) if dropout is not None else None
        self._use_self_loop = self_loop

        if weight:
            if self._use_basis:
                self.basis = dglnn.WeightBasis(
                    (in_feats, out_feats), num_bases, self._num_rels)
            else:
                self.weight = nn.Parameter(torch.Tensor(
                    self._num_rels, in_feats, out_feats))
                nn.init.xavier_uniform_(
                    self.weight, gain=nn.init.calculate_gain('relu'))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
            nn.init.zeros_(self.bias)

        if self_loop:
            self.self_loop_weight = nn.Parameter(
                torch.Tensor(in_feats, out_feats))
            nn.init.xavier_uniform_(
                self.self_loop_weight, gain=nn.init.calculate_gain('relu'))

    def _apply(
        self,
        ntype: str,
        inputs: torch.Tensor,
        inputs_dst: torch.Tensor = None,
    ) -> torch.Tensor:
        x = inputs

        if inputs_dst is not None:
            x += torch.matmul(inputs_dst[ntype], self.self_loop_weight)

        if self._use_bias:
            x += self.bias

        if self._activation is not None:
            x = self._activation(x)

        if self._dropout is not None:
            x = self._dropout(x)

        return x

    def forward(
        self,
        hg: dgl.DGLHeteroGraph,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        hg = hg.local_var()

        if self._use_weight:
            weight = self.basis() if self._use_basis else self.weight
            weight_dict = {self._rel_names[i]: {'weight': w.squeeze(
                dim=0)} for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            weight_dict = {}

        if self._use_self_loop:
            if hg.is_block:
                inputs_dst = {ntype: h[:hg.num_dst_nodes(
                    ntype)] for ntype, h in inputs.items()}
            else:
                inputs_dst = inputs
        else:
            inputs_dst = None

        x = self._conv(hg, inputs, mod_kwargs=weight_dict)
        x = {ntype: self._apply(ntype, h, inputs_dst)
             for ntype, h in x.items()}

        return x


class RelGraphEmbedding(nn.Module):
    def __init__(
        self,
        hg: dgl.DGLHeteroGraph,
        embedding_size: int,
        num_nodes: dict[str, int],
        node_feats: dict[str, torch.Tensor],
        node_feats_projection: bool,
    ):
        super().__init__()
        self._hg = hg
        self._node_feats = node_feats
        self._node_feats_projection = node_feats_projection
        self.node_embeddings = nn.ModuleDict()

        if node_feats_projection:
            self.embeddings = nn.ParameterDict()

        for ntype in hg.ntypes:
            if node_feats[ntype] is None:
                node_embedding = nn.Embedding(
                    num_nodes[ntype], embedding_size, sparse=True)
                nn.init.uniform_(node_embedding.weight, -1, 1)

                self.node_embeddings[ntype] = node_embedding
            elif node_feats[ntype] is not None and node_feats_projection:
                input_embedding_size = node_feats[ntype].shape[-1]
                embedding = nn.Parameter(torch.Tensor(
                    input_embedding_size, embedding_size))
                nn.init.xavier_uniform_(embedding)

                self.embeddings[ntype] = embedding

    def forward(
        self,
        in_nodes: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        x = {}

        for ntype, nid in in_nodes.items():
            if self._node_feats[ntype] is None:
                x[ntype] = self.node_embeddings[ntype](nid)
            else:
                if self._node_feats_projection:
                    x[ntype] = self._node_feats[ntype][nid] @ self.embeddings[ntype]
                else:
                    x[ntype] = self._node_feats[ntype][nid]

        return x

    def inference(self) -> dict[str, torch.Tensor]:
        x = {}

        for ntype in self._hg.ntypes:
            hg_nodes = self._hg.nodes(ntype)

            if self._node_feats[ntype] is None:
                x[ntype] = self.node_embeddings[ntype](hg_nodes)
            else:
                if self._node_feats_projection:
                    x[ntype] = self._node_feats[ntype] @ self.embeddings[ntype]
                else:
                    x[ntype] = self._node_feats[ntype]

        return x


class EntityClassify(nn.Module):
    def __init__(
        self,
        hg: dgl.DGLHeteroGraph,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        num_bases: int,
        num_layers: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
        dropout: float = None,
        self_loop: bool = False,
    ):
        super().__init__()
        self._hg = hg
        self._rel_names = sorted(list(set(hg.etypes)))
        self._num_rels = len(self._rel_names)

        if num_bases < 0 or num_bases > self._num_rels:
            self._num_bases = self._num_rels
        else:
            self._num_bases = num_bases

        self._layers = nn.ModuleList()

        self._layers.append(RelGraphConvLayer(
            in_feats,
            hidden_feats,
            self._rel_names,
            self._num_bases,
            activation=activation,
            dropout=dropout,
            self_loop=self_loop,
        ))

        for _ in range(1, num_layers - 1):
            self._layers.append(RelGraphConvLayer(
                hidden_feats,
                hidden_feats,
                self._rel_names,
                self._num_bases,
                activation=activation,
                dropout=dropout,
                self_loop=self_loop,
            ))

        self._layers.append(RelGraphConvLayer(
            hidden_feats,
            out_feats,
            self._rel_names,
            self._num_bases,
            activation=None,
            dropout=dropout,
            self_loop=self_loop,
        ))

    def forward(
        self,
        blocks: tuple[dgl.DGLHeteroGraph],
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        x = inputs

        for layer, block in zip(self._layers, blocks):
            x = layer(block, x)

        return x

    def inference(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        x = inputs

        for layer in self._layers:
            x = layer(self._hg, x)

        return x


def train(
    embedding_layer: nn.Module,
    model: nn.Module,
    device: Union[str, torch.device],
    embedding_optimizer: torch.optim.Optimizer,
    model_optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dataloader: dgl.dataloading.NodeDataLoader,
    labels: torch.Tensor,
    predict_category: str,
) -> tuple[float]:
    model.train()

    total_loss = 0
    total_accuracy = 0

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


def validate(
    embedding_layer: nn.Module,
    model: nn.Module,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mask: torch.Tensor,
    labels: torch.Tensor,
    predict_category: str,
) -> tuple[float]:
    model.eval()

    start = default_timer()

    with torch.no_grad():
        embedding = embedding_layer.inference()
        logits = model.inference(embedding)[predict_category]
        loss = loss_function(logits[mask], labels[mask])

        _, indices = torch.max(logits[mask], dim=1)
        correct = torch.sum(indices == labels[mask])
        accuracy = correct.item() / len(labels[mask])

    stop = default_timer()
    time = stop - start

    return time, loss, accuracy
