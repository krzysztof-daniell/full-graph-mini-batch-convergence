from typing import Callable, Union

import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: Union[int, list[int]],
        out_feats: int,
        num_layers: int,
        aggregator_type: str = 'mean',
        batch_norm: bool = False,
        input_dropout: float = 0,
        dropout: float = 0,
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__()
        self._num_layers = num_layers
        self._input_dropout = nn.Dropout(input_dropout)
        self._dropout = nn.Dropout(dropout)
        self._activation = activation

        if isinstance(hidden_feats, int):
            self._hidden_feats = [hidden_feats for _ in range(num_layers - 1)]
        else:
            self._hidden_feats = hidden_feats

        self._layers = nn.ModuleList()

        self._layers.append(dglnn.SAGEConv(
            in_feats, self._hidden_feats[0], aggregator_type))

        for i in range(1, num_layers - 1):
            self._layers.append(dglnn.SAGEConv(self._hidden_feats[i - 1],
                                self._hidden_feats[i], aggregator_type))

        self._layers.append(dglnn.SAGEConv(
            self._hidden_feats[-1], out_feats, aggregator_type))

        if batch_norm:
            self._batch_norms = nn.ModuleList()

            for i in range(num_layers - 1):
                self._batch_norms.append(nn.BatchNorm1d(self._hidden_feats[i]))

        else:
            self._batch_norms = None

    def _apply_layers(
        self,
        layer_idx: int,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = inputs

        if self._batch_norms is not None:
            x = self._batch_norms[layer_idx](x)

        if self._activation is not None:
            x = self._activation(x)

        x = self._dropout(x)

        return x

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
                    x = self._apply_layers(i, x)
        else:
            for i, layer in enumerate(self._layers):
                x = layer(g, x)

                if i < self._num_layers - 1:
                    x = self._apply_layers(i, x)

        return x
