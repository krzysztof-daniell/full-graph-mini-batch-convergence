from collections.abc import Callable
from timeit import default_timer
from typing import Union

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


def train(
    model: nn.Module,
    device: Union[str, torch.device],
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dataloader: dgl.dataloading.NodeDataLoader,
) -> tuple[float]:
    model.train()

    total_loss = 0
    total_accuracy = 0

    start = default_timer()

    for step, (_, _, blocks) in enumerate(dataloader):
        optimizer.zero_grad()

        blocks = [block.int().to(device) for block in blocks]

        inputs = blocks[0].srcdata['feat']
        labels = blocks[-1].dstdata['label']

        logits = model(blocks, inputs)
        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()

        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        accuracy = correct.item() / len(labels)

        total_loss += loss.item()
        total_accuracy += accuracy

    stop = default_timer()
    time = stop - start

    total_loss /= step
    total_accuracy /= step

    return time, total_loss, total_accuracy


def validate(
    model: nn.Module,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    g: dgl.DGLGraph,
    mask: torch.Tensor,
) -> tuple[float]:
    features = g.ndata['feat']
    labels = g.ndata['label']

    model.eval()

    start = default_timer()

    with torch.no_grad():
        logits = model.inference(g, features)
        loss = loss_function(logits[mask], labels[mask])

        _, indices = torch.max(logits[mask], dim=1)
        correct = torch.sum(indices == labels[mask])
        accuracy = correct.item() / len(labels[mask])

    stop = default_timer()
    time = stop - start

    return time, loss, accuracy
