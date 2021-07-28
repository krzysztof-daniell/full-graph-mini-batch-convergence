from collections.abc import Callable

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
        self._self_loop = self_loop

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
        if inputs_dst is not None:
            x = inputs + \
                torch.matmul(inputs_dst[ntype], self.self_loop_weight)

        if self._use_bias:
            x = x + self.bias

        if self._activation is not None:
            x = self._activation(x)

        if self._dropout is not None:
            x = self._dropout

        return x

    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        g = g.local_var()

        if self._use_weight:
            weight = self.basis if self._use_basis else self.weight
            weight_dict = {self._rel_names[i]: w.squeeze(
                dim=1) for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            weight_dict = {}

        if self._self_loop is not None:
            if g.is_block():
                inputs_dst = {ntype: h[:g.num_dst_nodes(
                    ntype)] for ntype, h in inputs.items()}
            else:
                inputs_dst = inputs
        else:
            inputs_dst = None

        x = self._conv(g, inputs, mod_kwargs=weight_dict)
        x = {ntype: self._apply(ntype, h, inputs_dst)
             for ntype, h in x.items()}

        return x


# class RelGraphEmbedding(nn.Module):
#     """This one is from benchmarks"""
#     def __init__(
#         self,
#         g: dgl.DGLHeteroGraph,
#         embedding_size: int,
#         num_nodes: int,
#         node_feats: dict[str, torch.Tensor],
#         # embedding_name: str = 'embedding',  # TODO: this is in examples, but it's not used
#         # activation: Callable[[torch.Tensor], torch.Tensor] = None,
#         # dropout: float = None,
#     ):
#         super().__init__()
#         self._g = g
#         # self._embedding_size = embedding_size
#         # self._num_nodes = num_nodes
#         self._node_feats = node_feats
#         # self._embedding_name = embedding_name
#         # self._activation = activation
#         # self._dropout = nn.Dropout(dropout) if dropout is not None else None

#         self._embeddings = nn.ParameterDict()
#         self._node_embeddings = nn.ModuleDict()

#         for ntype in g.ntypes:
#             if node_feats[ntype] is None:
#                 sparse_embedding = torch.nn.Embedding(
#                     num_nodes[ntype], embedding_size, sparse=True)
#                 # TODO: needs floats as args?
#                 nn.init.uniform_(sparse_embedding.weight, -1, 1)

#                 self._node_embeddings[ntype] = sparse_embedding
#             else:
#                 input_embedding_size = node_feats[ntype].shape[-1]
#                 embedding = nn.Parameter(torch.Tensor(
#                     input_embedding_size, embedding_size))
#                 nn.init.xavier_uniform_(embedding)

#                 self._embeddings[ntype] = embedding

#     def forward(
#         self,
#         block: dgl.DGLHeteroGraph = None,
#     ) -> dict[str, torch.Tensor]:
#         x = {}

#         if block is not None:
#             for ntype in block.ntypes:
#                 if self._node_feats[ntype] is None:
#                     x[ntype] = self._node_embeddings[ntype][block.nodes(ntype)]
#                 else:
#                     x[ntype] = self._node_feats[ntype][block.nodes(
#                         ntype)] @ self._embeddings[ntype][block.nodes(ntype)]  # TODO: examples = self._embeddings[ntype]
#         else:
#             for ntype in self._g.ntypes:
#                 if self._node_feats[ntype] is None:
#                     x[ntype] = self._node_embeddings[ntype]
#                 else:
#                     x[ntype] = self._node_feats[ntype] @ self._embeddings[ntype]

#         return x


class RelGraphEmbedding(nn.Module):
    """This one is from examples"""

    def __init__(
        self,
        g: dgl.DGLHeteroGraph,
        embedding_size: int,
    ):
        super().__init__()
        self._g = g
        self._embeddings = nn.ParameterDict()

        for ntype in g.ntypes:
            embedding = nn.Parameter(torch.Tensor(
                g.num_nodes(ntype), embedding_size))
            nn.init.xavier_uniform_(
                embedding, gain=nn.init.calculate_gain('relu'))

            self._embeddings[ntype] = embedding

    def forward(self, block: dgl.DGLHeteroGraph) -> dict[str, torch.Tensor]:
        x = {}

        for ntype in block.ntypes:
            x[ntype] = self._embeddings[ntype][block.nodes(ntype)]

        return x

    def inference(self):
        return self._embeddings


class EntityClassify(nn.Module):
    def __init__(
        self,
        g: dgl.DGLHeteroGraph,
        hidden_feats: int,
        out_feats: int,
        num_bases: int,
        num_layers: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
        dropout: float = None,
        self_loop: bool = False,
    ):
        super().__init__()
        self._g = g
        self._rel_names = sorted(list(set(g.etypes)))
        self._num_rels = len(self._rel_names)

        if num_bases < 0 or num_bases > self._num_rels:
            self._num_bases = self._num_bases
        else:
            self._num_bases = num_bases

        self._embedding_layer = RelGraphEmbedding(g, hidden_feats)
        self._layers = nn.ModuleList()

        self._layers.append(RelGraphConvLayer(
            hidden_feats,
            hidden_feats,
            self._rel_names,
            self._num_bases,
            weight=False,
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
    ) -> dict[str, torch.Tensor]:
        for layer, block in zip(self._layers, blocks):
            x = self._embedding_layer(block)
            x = layer(block, x)

        return x

    def inference(self):
        x = self._embedding_layer.inference()

        for layer in self._layers:
            x = layer(self._g, x)

        return x
