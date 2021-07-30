from collections.abc import Callable
from timeit import default_timer
from typing import Union

import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn


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
        blocks: dgl.DGLHeteroGraph, 
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = inputs

        for layer, block in zip(self._layers, blocks):
            x = layer(block, x, block.edata['etype'], block.edata['norm'])

        return x


class RelGraphEmbedLayer(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_ntypes: int,
        input_size: list[Union[int, None]],
        embedding_size: int,
    ):
        super().__init__()
        self._num_ntypes = num_ntypes
        self._embedding_size = embedding_size

        self._embeddings = nn.ParameterDict()
        self._node_embeddings = {}

        for ntype in range(num_ntypes):  # TODO: make logic from benchmarks
            if isinstance(input_size[ntype], int):
                # self._node_embeddings[f'{ntype}'] = dglnn.NodeEmbedding(
                #     input_size[ntype], 
                #     embedding_size, 
                #     name=f'{ntype}', 
                #     init_func=self._initializer,
                # )
                node_embedding = nn.Embedding(input_size[ntype], embedding_size, sparse=True)
                nn.init.uniform_(node_embedding.weight, -1, 1)

                self._node_embeddings[f'{ntype}'] = node_embedding
            else:
                embedding = nn.Parameter(torch.empty(input_size[ntype].shape[-1], embedding_size))
                nn.init.xavier_uniform_(embedding)

                self._embeddings[f'{ntype}'] = embedding

    def _initializer(embedding: torch.Tensor):
        embedding.uniform_(-1, 1)

        return embedding

    def forward(
        self, 
        node_ids: torch.Tensor, 
        ntype_ids: torch.Tensor, 
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = torch.empty(node_ids.shape[0], self._embedding_size)
        locs = [None for _ in range(self._num_ntypes)]

        for ntype in range(self._num_ntypes):
            locs[ntype] = (ntype_ids == ntype).nonzero().squeeze(dim=-1)
            
        for ntype in range(self._num_ntypes):
            loc = ntype_ids == ntype

            if isinstance(inputs[ntype], int):
                embeddings[loc] = self._node_embeddings[f'{ntype}']()



