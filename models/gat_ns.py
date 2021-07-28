from collections.abc import Callable

import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn


class GAT(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        heads: int,
        num_layers: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        feat_drop: float = 0,
        attn_drop: float = 0,
        negative_slope: float = 0.2,
        residual: bool = False,
    ):
        super().__init__()
        self._num_layers = num_layers
        self._layers = nn.ModuleList()

        self._layers.append(dglnn.GATConv(
            in_feats,
            hidden_feats,
            num_heads=heads[0],
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=False,
            activation=activation,
        ))

        for i in range(1, num_layers - 1):
            self._layers.append(dglnn.GATConv(
                hidden_feats * heads[i - 1],
                hidden_feats,
                num_heads=heads[i],
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                activation=activation,
            ))

        self._layers.append(dglnn.GATConv(
            hidden_feats * heads[-2],
            out_feats,
            num_heads=heads[-1],
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            activation=None,
        ))

    def forward(
        self,
        blocks: tuple[dgl.DGLGraph],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = inputs

        for i, (layer, block) in enumerate(zip(self._layers, blocks)):
            x = layer(block, x)

            if i < self._num_layers - 1:
                x = x.flatten(start_dim=1)

        x = x.mean(dim=1)

        return x

    def inference(
        self,
        g: dgl.DGLGraph,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = inputs

        for i, layer in enumerate(self._layers):
            x = layer(g, x)

            if i < self._num_layers - 1:
                x = x.flatten(start_dim=1)

        x = x.mean(dim=1)

        return x
