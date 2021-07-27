from collections.abc import Callable

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
        activation: Callable[[torch.Tensor], torch.Tensor],
        dropout: float = None,
    ):
        super().__init__()
        self._hidden_feats = hidden_feats
        self._out_feats = out_feats
        self._num_layers = num_layers
        self._dropout = nn.Dropout(dropout) if dropout is not None else None
        self._activation = activation
        self._layers = nn.ModuleList()
        
        self._layers.append(dglnn.SAGEConv(in_feats, hidden_feats, 'mean'))

        for _ in range(1, num_layers - 1):
            self._layers.append(dglnn.SAGEConv(
                hidden_feats, hidden_feats, 'mean'))

        self._layers.append(dglnn.SAGEConv(hidden_feats, out_feats, 'mean'))

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

                if self._dropout is not None:
                    x = self._dropout(x)

        return x
