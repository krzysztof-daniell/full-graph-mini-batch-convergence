from collections.abc import Callable
from typing import Union

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair


class GATConv(nn.Module):
    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        node_out_feats: int,
        edge_out_feats: int,
        num_heads: int,
        norm: str = 'none',
        attn_dropout: float = 0,
        edge_dropout: float = 0,
        negative_slope: float = 0.2,
        residual: bool = True,
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
        use_attn_dst: bool = True,
        allow_zero_in_degree: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self._node_out_feats = node_out_feats
        self._edge_out_feats = edge_out_feats
        self._num_heads = num_heads
        self._norm = norm
        self._attn_dropout = nn.Dropout(attn_dropout)
        self._edge_dropout = edge_dropout
        self._leaky_relu = nn.LeakyReLU(negative_slope)
        self._activation = activation
        self._allow_zero_in_degree = allow_zero_in_degree
        self._in_src_feats, self._in_dst_feats = expand_as_pair(node_in_feats)

        if isinstance(node_in_feats, tuple):
            self._fc_src = nn.Linear(
                self._in_src_feats, num_heads * node_out_feats, bias=False)
            self._fc_dst = nn.Linear(
                self._in_dst_feats, num_heads * node_out_feats, bias=False)
        else:
            self._fc = nn.Linear(self._in_src_feats,
                                 num_heads * node_out_feats, bias=False)

        self._attn_src = nn.Parameter(
            torch.Tensor(1, num_heads, node_out_feats))

        if use_attn_dst:
            self._attn_dst = nn.Parameter(torch.Tensor(
                1, num_heads, node_out_feats))
        else:
            self._attn_dst = None

        if edge_in_feats > 0:
            self._fc_edge = nn.Linear(
                edge_in_feats, num_heads * edge_out_feats, bias=False)
            self._attn_edge = nn.Parameter(torch.Tensor(
                1, num_heads, edge_out_feats))
        else:
            self.register_buffer('_fc_edge', None)
            self.register_buffer('_attn_edge', None)

        if residual:
            self._fc_res = nn.Linear(
                self._in_dst_feats, num_heads * node_out_feats, bias=False)
        else:
            self.register_buffer('_fc_res', None)

        if bias:
            self._bias = nn.Parameter(torch.Tensor(num_heads * node_out_feats))
        else:
            self.register_buffer('_bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')

        if hasattr(self, '_fc'):
            nn.init.xavier_normal_(self._fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self._fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self._fc_dst.weight, gain=gain)

        nn.init.xavier_normal_(self._attn_src, gain=gain)

        if self._attn_dst is not None:
            nn.init.xavier_normal_(self._attn_dst, gain=gain)

        if self._fc_edge is not None:
            nn.init.xavier_normal_(self._fc_edge.weight, gain=gain)

        if self._attn_edge is not None:
            nn.init.xavier_normal_(self._attn_edge, gain=gain)

        if self._fc_res is not None:
            nn.init.xavier_normal_(self._fc_res.weight, gain=gain)

        if self._bias is not None:
            nn.init.zeros_(self._bias)

    def set_allow_zero_in_degree(self, value: bool):
        self._allow_zero_in_degree = value

    def forward(
        self,
        g: dgl.DGLGraph,
        node_inputs: torch.Tensor,
        edge_inputs: torch.Tensor = None,
    ) -> torch.Tensor:
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    assert False

        if isinstance(node_inputs, tuple):
            node_inputs_src = node_inputs[0]
            node_inputs_dst = node_inputs[1]

            if not hasattr('_fc_src'):
                feat_fc_src = self._fc(node_inputs_src).view(
                    -1, self._num_heads, self._node_out_feats)
                feat_fc_dst = self._fc(node_inputs_dst).view(
                    -1, self._num_heads, self._node_out_feats)
            else:
                feat_fc_src = self._fc_src(node_inputs_src).view(
                    -1, self._num_heads, self._node_out_feats)
                feat_fc_dst = self._fc_dst(node_inputs_dst).view(
                    -1, self._num_heads, self._node_out_feats)
        else:
            node_inputs_dst = node_inputs

            feat_fc_src = self._fc(node_inputs).view(
                -1, self._num_heads, self._node_out_feats)

            if g.is_block:
                node_inputs_dst = node_inputs_dst[:g.num_dst_nodes()]

                feat_fc_dst = feat_fc_src[:g.num_dst_nodes()]
            else:
                feat_fc_dst = feat_fc_src

        if self._norm in ['both', 'left']:
            degrees = g.out_degrees().float().clamp(min=1)

            if self._norm == 'both':
                norm = torch.pow(degrees, -0.5)
            else:
                norm = 1 / degrees

            shape = norm.shape + (1,) * (feat_fc_src.dim() - 1)
            norm = torch.reshape(norm, shape)

            feat_fc_src *= norm

        attn_src = (feat_fc_src * self._attn_src).sum(dim=-1).unsqueeze(-1)
        g.srcdata.update({'feat_fc_src': feat_fc_src, 'attn_src': attn_src})

        if self._attn_dst is not None:
            attn_dst = (feat_fc_dst * self._attn_dst).sum(dim=-1).unsqueeze(-1)
            g.dstdata.update({'attn_dst': attn_dst})

            g.apply_edges(fn.u_add_v('attn_src', 'attn_dst', 'attn_node'))
        else:
            g.apply_edges(fn.copy_u('attn_src', 'attn_node'))

        e = g.edata['attn_node']

        if edge_inputs is not None:
            feat_fc_edge = self._fc_edge(edge_inputs).view(
                -1, self._num_heads, self._edge_out_feats)
            attn_edge = (feat_fc_edge *
                         self._attn_edge).sum(dim=-1).unsqueeze(-1)

            g.edata.update({'attn_edge': attn_edge})
            e += g.edata['attn_edge']

        e = self._leaky_relu(e)

        if self.training and self._edge_dropout > 0:
            perm = torch.randperm(g.num_edges(), device=e.device)
            bound = int(g.num_edges() * self._edge_dropout)
            eids = perm[bound:]

            g.edata['attn'] = torch.zeros_like(e)
            g.edata['attn'][eids] = self._attn_dropout(edge_softmax(
                g, e[eids], eids=eids))
        else:
            g.edata['attn'] = self._attn_dropout(edge_softmax(g, e))

        g.update_all(fn.u_mul_e('feat_fc_src', 'attn', 'msg'),
                     fn.sum('msg', 'feat_fc_src'))
        x = g.dstdata['feat_fc_src']

        if self._norm in ['both', 'right']:
            degrees = g.in_degrees().float().clamp(min=1)

            if self._norm == 'both':
                norm = torch.pow(degrees, -0.5)
            else:
                norm = 1 / degrees

            shape = norm.shape + (1,) * (feat_fc_dst.dim() - 1)
            norm = torch.reshape(norm, shape)

            x *= norm

        if self._fc_res is not None:
            x += self._fc_res(node_inputs_dst).view(
                node_inputs_dst.shape[0], -1, self._node_out_feats)

        if self._bias is not None:
            x += self._bias.view(-1, self._node_out_feats)

        if self._activation is not None:
            x = self._activation(x)

        return x


class GAT(nn.Module):
    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        node_hidden_feats: int,
        edge_hidden_feats: int,
        out_feats: int,
        num_heads: int,
        num_layers: int,
        norm: str = 'none',
        batch_norm: bool = False,
        input_dropout: float = 0,
        attn_dropout: float = 0,
        edge_dropout: float = 0,
        dropout: float = 0,
        negative_slope: float = 0.2,
        residual: bool = True,
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
        use_attn_dst: bool = True,
        allow_zero_in_degree: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self._edge_in_feats = edge_in_feats
        self._num_layers = num_layers
        self._input_dropout = nn.Dropout(input_dropout)
        self._dropout = nn.Dropout(dropout)
        self._residual = residual
        self._activation = activation

        self._layers = nn.ModuleList()

        self._layers.append(GATConv(
            node_in_feats,
            edge_in_feats,
            node_hidden_feats,
            edge_hidden_feats,
            num_heads,
            norm=norm,
            attn_dropout=attn_dropout,
            edge_dropout=edge_dropout,
            negative_slope=negative_slope,
            residual=residual,
            use_attn_dst=use_attn_dst,
            allow_zero_in_degree=allow_zero_in_degree,
            bias=bias,
        ))

        for _ in range(1, num_layers - 1):
            self._layers.append(GATConv(
                num_heads * node_hidden_feats,
                edge_in_feats,
                node_hidden_feats,
                edge_hidden_feats,
                num_heads,
                norm=norm,
                attn_dropout=attn_dropout,
                edge_dropout=edge_dropout,
                negative_slope=negative_slope,
                residual=residual,
                use_attn_dst=use_attn_dst,
                allow_zero_in_degree=allow_zero_in_degree,
                bias=bias,
            ))

        self._layers.append(GATConv(
            num_heads * node_hidden_feats,
            edge_in_feats,
            out_feats,
            edge_hidden_feats,
            num_heads,
            norm=norm,
            attn_dropout=attn_dropout,
            edge_dropout=edge_dropout,
            negative_slope=negative_slope,
            residual=residual,
            use_attn_dst=use_attn_dst,
            allow_zero_in_degree=allow_zero_in_degree,
            bias=bias,
        ))

        if batch_norm:
            self._batch_norms = nn.ModuleList()

            for _ in range(num_layers - 1):
                self._batch_norms.append(nn.BatchNorm1d(
                    num_heads * node_hidden_feats))
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
                if self._edge_in_feats > 0:
                    edge_inputs = block.edata['feat']
                else:
                    edge_inputs = None

                x = layer(block, x, edge_inputs)

                if i < self._num_layers - 1:
                    x = self._apply_layers(i, x)
        else:
            for i, layer in enumerate(self._layers):
                if self._edge_in_feats > 0:
                    edge_inputs = g.edata['feat']
                else:
                    edge_inputs = None

                x = layer(g, x, edge_inputs)

                if i < self._num_layers - 1:
                    x = self._apply_layers(i, x)

        x = x.mean(dim=-2)

        return x
