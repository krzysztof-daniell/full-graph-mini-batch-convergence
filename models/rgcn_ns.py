from collections.abc import Callable

import dgl
import dgl.nn.pytorch as dglnn
import torch
from torch._C import InferredType
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
        self_loop: bool = False,
        dropout: float = None,
    ):
        super().__init__()
        self._num_rels = len(rel_names)
        self._conv = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(
            in_feats, out_feats, norm='right', weight=False, bias=False) for rel in rel_names})
        self._use_weight = weight
        self._use_basis = num_bases < self._num_rels and weight
        self._use_bias = bias
        self._self_loop = self_loop
        self._activation = activation
        self._dropout = nn.Dropout(dropout) if dropout is not None else None

        if weight:
            if self._use_basis:
                self._basis = dglnn.WeightBasis(
                    (in_feats, out_feats), num_bases, self._num_rels)
            else:
                self._weight = nn.Parameter(torch.Tensor(
                    self._num_rels, in_feats, out_feats))
                nn.init.xavier_uniform_(
                    self._weight, gain=nn.init.calculate_gain('relu'))

        if bias:
            self._bias = nn.Parameter(torch.Tensor(out_feats))
            nn.init.zeros_(self._bias)

        if self_loop:
            self._self_loop_weight = nn.Parameter(
                torch.Tensor(in_feats, out_feats))
            nn.init.xavier_uniform_(
                self._self_loop_weight, gain=nn.init.calculate_gain('relu'))


class RGCNNodeSampling(nn.Module):
    def __init__(self):
        super().__init__()
        pass
