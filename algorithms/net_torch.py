# -*- coding: utf-8 -*-
# Implementation of Neural Networks in PyTorch
# @Time    : 2022/8/19
# @Author  : Xu Hongzuo


import numpy as np
import torch
import torch_geometric
from torch.nn import functional as F


def choose_net(network_name):
    if network_name == 'mlp':
        return MLPnet
    elif network_name == 'gru':
        return GRUNet
    elif network_name == 'lstm':
        return LSTMNet
    elif network_name == 'gin':
        return GinEncoderGraph
    else:
        raise NotImplementedError("")


def choose_act_func(activation):
    if activation == 'relu':
        act_module = torch.nn.ReLU()
        act_f = torch.relu
    elif activation == 'leaky_relu':
        act_module = torch.nn.LeakyReLU()
        act_f = F.leaky_relu
    elif activation == 'tanh':
        act_module = torch.nn.Tanh()
        act_f = torch.tanh
    elif activation == 'sigmoid':
        act_module = torch.nn.Sigmoid()
        act_f = torch.sigmoid
    else:
        raise NotImplementedError('')
    return act_module, act_f


def choose_pooling_func(pooling):
    if pooling == 'sum':
        pool_f = torch_geometric.nn.global_add_pool
    elif pooling == 'mean':
        pool_f = torch_geometric.nn.global_mean_pool
    elif pooling == 'max':
        pool_f = torch_geometric.nn.global_max_pool
    else:
        raise NotImplementedError('')
    return pool_f


class MLPnet(torch.nn.Module):
    def __init__(self, n_features, n_hidden=[500, 100], n_emb=20, activation='tanh',
                 skip_connection=None, dropout=None, be_size=None):
        super(MLPnet, self).__init__()
        self.skip_connection = skip_connection
        self.n_emb = n_emb

        assert activation in ['relu', 'tanh', 'sigmoid', 'leaky_relu']

        if type(n_hidden)==int: n_hidden = [n_hidden]
        if type(n_hidden)==str: n_hidden = n_hidden.split(','); n_hidden = [int(a) for a in n_hidden]
        num_layers = len(n_hidden)

        self.be_size = be_size

        self.layers = []
        for i in range(num_layers+1):
            in_channels, out_channels = self.get_in_out_channels(i, num_layers, n_features,
                                                                 n_hidden, n_emb, skip_connection)
            self.layers += [LinearBlock(in_channels, out_channels,
                                        activation=activation if i != num_layers else None,
                                        skip_connection=skip_connection if i != num_layers else 0,
                                        dropout=dropout,
                                        be_size=be_size)]
        self.network = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        if self.be_size is not None:
            x = x.repeat(self.be_size, 1)
        x = self.network(x)
        return x

    def get_in_out_channels(self, i, num_layers, n_features, n_hidden, n_emb, skip_connection):
        if skip_connection is None:
            in_channels = n_features if i == 0 else n_hidden[i-1]
            out_channels = n_emb if i == num_layers else n_hidden[i]
        elif skip_connection == 'concat':
            in_channels = n_features if i == 0 else np.sum(n_hidden[:i])+n_features
            out_channels = n_emb if i == num_layers else n_hidden[i]
        else:
            raise NotImplementedError('')
        return in_channels, out_channels


class AEnet(torch.nn.Module):
    def __init__(self, n_features, n_hidden=[500, 100], n_emb=20, activation='tanh',
                 skip_connection=None, be_size=None):
        super(AEnet, self).__init__()
        assert activation in ['tanh', 'relu']
        if type(n_hidden)==int: n_hidden = [n_hidden]
        if type(n_hidden)==str: n_hidden = n_hidden.split(','); n_hidden = [int(a) for a in n_hidden]
        num_layers = len(n_hidden)
        self.be_size = be_size

        self.encoder_layers = []
        for i in range(num_layers+1):
            in_channels = n_features if i == 0 else n_hidden[i-1]
            out_channels = n_emb if i == num_layers else n_hidden[i]
            self.encoder_layers += [LinearBlock(in_channels, out_channels,
                                                bias=False,
                                                activation=activation if i != num_layers else None,
                                                skip_connection=None,
                                                be_size=be_size)]

        self.decoder_layers = []
        for i in range(num_layers+1):
            in_channels = n_emb if i == 0 else n_hidden[num_layers-i]
            out_channels = n_features if i == num_layers else n_hidden[num_layers-1-i]
            self.decoder_layers += [LinearBlock(in_channels, out_channels,
                                                bias=False,
                                                activation=activation if i != num_layers else None,
                                                skip_connection=None,
                                                be_size=be_size)]

        self.encoder = torch.nn.Sequential(*self.encoder_layers)
        self.decoder = torch.nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        if self.be_size is not None:
            x = x.repeat(self.be_size, 1)
        enc = self.encoder(x)
        xx = self.decoder(enc)

        return enc, xx, x


class LinearBlock(torch.nn.Module):
    """Linear layer with support to concatenation-based skip connection and batch ensemble"""
    def __init__(self, in_channels, out_channels,
                 bias=False, activation='tanh',
                 skip_connection=None, dropout=None, be_size=None):
        super(LinearBlock, self).__init__()

        self.act = activation
        self.skip_connection = skip_connection
        self.dropout = dropout
        self.be_size = be_size

        if activation is not None:
            self.act_layer, _ = choose_act_func(activation)

        if dropout is not None:
            self.dropout_layer = torch.nn.Dropout(p=dropout)

        if be_size is not None:
            bias = False
            self.ri = torch.nn.Parameter(torch.randn(be_size, in_channels))
            self.si = torch.nn.Parameter(torch.randn(be_size, out_channels))

        self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        if self.be_size is not None:
            R = torch.repeat_interleave(self.ri, int(x.shape[0]/self.be_size), dim=0)
            S = torch.repeat_interleave(self.si, int(x.shape[0]/self.be_size), dim=0)

            x1 = torch.mul(self.linear(torch.mul(x, R)), S)
        else:
            x1 = self.linear(x)

        if self.act is not None:
            x1 = self.act_layer(x1)

        if self.dropout is not None:
            x1 = self.dropout_layer(x1)

        if self.skip_connection == 'concat':
            x1 = torch.cat([x, x1], axis=1)

        return x1


class GRUNet(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=20, layers=1):
        super(GRUNet, self).__init__()
        self.gru = torch.nn.GRU(n_features, hidden_size=hidden_dim, batch_first=True, num_layers=layers)

    def forward(self, x):
        _, hn = self.gru(x)
        return hn[-1]


class LSTMNet(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=20, layers=1, bidirectional=False):
        super(LSTMNet, self).__init__()
        self.bi = bidirectional
        self.lstm = torch.nn.LSTM(n_features, hidden_size=hidden_dim, batch_first=True,
                                  bidirectional=bidirectional, num_layers=layers)

    def forward(self, x):
        output, (hn, c) = self.lstm(x)
        emb = hn[-1]
        if self.bi:
            emb = torch.cat([hn[-2], hn[-1]], dim=1)
        return emb


class GinEncoderGraph(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_emb, n_layers,
                 pooling='sum', activation='relu'):
        super(GinEncoderGraph, self).__init__()

        assert pooling in ['sum', 'mean', 'max']
        assert activation in ['relu', 'tanh', 'sigmoid', 'leaky_relu']

        self.pooling = pooling
        self.num_gc_layers = n_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.act_module, self.act_f = choose_act_func(activation)
        self.pool_f = choose_pooling_func(pooling)

        for i in range(self.num_gc_layers):
            in_channel = n_features if i == 0 else n_hidden
            out_channel = n_emb if i == n_layers-1 else n_hidden
            nn = torch.nn.Sequential(
                torch.nn.Linear(in_channel, n_hidden, bias=False),
                self.act_module,
                torch.nn.Linear(n_hidden, out_channel, bias=False)
            )
            conv = torch_geometric.nn.GINConv(nn)
            bn = torch.nn.BatchNorm1d(out_channel)
            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_gc_layers-1):
            x = self.act_f(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        x = self.convs[-1](x, edge_index)
        xs.append(x)

        xpool = self.pool_f(xs[-1], batch)

        return xpool, torch.cat(xs, 1)
