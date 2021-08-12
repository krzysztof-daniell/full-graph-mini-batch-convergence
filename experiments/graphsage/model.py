from collections.abc import Callable
from typing import Union

import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        num_layers: int,
        aggregator_type: str,
        batch_norm: bool,
        activation: Callable[[torch.Tensor], torch.Tensor],
        input_dropout: float,
        dropout: float,
    ):
        super().__init__()
        self._hidden_feats = hidden_feats
        self._out_feats = out_feats
        self._num_layers = num_layers
        self._input_dropout = nn.Dropout(input_dropout)
        self._dropout = nn.Dropout(dropout)
        self._activation = activation

        self._layers = nn.ModuleList()

        self._layers.append(dglnn.SAGEConv(
            in_feats, hidden_feats, aggregator_type))

        for _ in range(1, num_layers - 1):
            self._layers.append(dglnn.SAGEConv(
                hidden_feats, hidden_feats, aggregator_type))

        self._layers.append(dglnn.SAGEConv(
            hidden_feats, out_feats, aggregator_type))

        if batch_norm:
            self._batch_norms = nn.ModuleList()

            for _ in range(num_layers - 1):
                self._batch_norms.append(nn.BatchNorm1d(hidden_feats))
        else:
            self._batch_norms = None

    def forward(
        self,
        g: Union[dgl.DGLGraph, list[dgl.DGLGraph]],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = self._input_dropout(inputs)

        if isinstance(g, list):
            for i, (layer, block) in enumerate(zip(self._layers, g)):
                x = layer(block, x)

                if i < self._num_layers - 1:
                    if self._batch_norms is not None:
                        x = self._batch_norms[i](x)

                    x = self._activation(x)
                    x = self._dropout(x)
        else:
            for i, layer in enumerate(self._layers):
                x = layer(g, x)

                if i < self._num_layers - 1:
                    if self._batch_norms is not None:
                        x = self._batch_norms[i](x)

                    x = self._activation(x)
                    x = self._dropout(x)

        return x
