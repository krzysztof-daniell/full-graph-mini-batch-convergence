from typing import Callable, Union

import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GATv2Conv


class GATv2(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        num_heads: int,
        num_layers: int,
        batch_norm: bool = False,
        input_dropout: float = 0,
        attn_dropout: float = 0,
        dropout: float = 0,
        negative_slope: float = 0.2,
        residual: bool = True,
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
        allow_zero_in_degree: bool = True,
        bias: bool = True,
        share_weights: bool = False,
    ):
        super().__init__()
        self._num_layers = num_layers
        self._input_dropout = nn.Dropout(input_dropout)
        self._dropout = nn.Dropout(dropout)
        self._residual = residual
        self._activation = activation

        self._layers = nn.ModuleList()

        self._layers.append(GATv2Conv(
            in_feats,
            hidden_feats,
            num_heads,
            attn_drop=attn_dropout,
            negative_slope=negative_slope,
            residual=residual,
            allow_zero_in_degree=allow_zero_in_degree,
            bias=bias,
            share_weights=share_weights,
        ))

        for _ in range(1, num_layers - 1):
            self._layers.append(GATv2Conv(
                num_heads * hidden_feats,
                hidden_feats,
                num_heads,
                attn_drop=attn_dropout,
                negative_slope=negative_slope,
                residual=residual,
                allow_zero_in_degree=allow_zero_in_degree,
                bias=bias,
                share_weights=share_weights,
            ))

        self._layers.append(GATv2Conv(
            num_heads * hidden_feats,
            out_feats,
            num_heads,
            attn_drop=attn_dropout,
            negative_slope=negative_slope,
            residual=residual,
            allow_zero_in_degree=allow_zero_in_degree,
            bias=bias,
            share_weights=share_weights,
        ))

        if batch_norm:
            self._batch_norms = nn.ModuleList()

            for _ in range(num_layers - 1):
                self._batch_norms.append(nn.BatchNorm1d(
                    num_heads * hidden_feats))
        else:
            self._batch_norms = None

    def _apply_layers(
        self,
        layer_idx: int,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = inputs.flatten(1, -1)

        if self._batch_norms is not None:
            x = self._batch_norms[layer_idx](x)

        if self._activation is not None:
            x = self._activation(x, inplace=True)

        x = self._dropout(x)

        return x

    def forward(
        self,
        g: Union[dgl.DGLGraph, list[dgl.DGLGraph]],
        node_inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = self._input_dropout(node_inputs)

        if isinstance(g, list):
            for i, (block, layer) in enumerate(zip(g, self._layers)):
                x = layer(block, x)

                if i < self._num_layers - 1:
                    x = self._apply_layers(i, x)
        else:
            for i, layer in enumerate(self._layers):
                x = layer(g, x)

                if i < self._num_layers - 1:
                    x = self._apply_layers(i, x)

        x = x.mean(dim=-2)

        return x
