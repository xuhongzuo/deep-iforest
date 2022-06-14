import torch
import torch_geometric
from torch.nn import functional as F


class GinEncoderGraph(torch.nn.Module):
    def __init__(self, n_features, hidden_dim, layers, embedding_dim,
                 pooling='sum', activation='relu'):
        super(GinEncoderGraph, self).__init__()
        import torch_geometric

        assert pooling in ['sum', 'mean', 'max']
        assert activation in ['relu', 'tanh', 'sigmoid', 'leaky_relu']

        self.pooling = pooling
        self.num_gc_layers = layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        if activation == 'relu':
            self.act_module = torch.nn.ReLU()
            self.act_f = torch.relu
        elif activation == 'leaky_relu':
            self.act_module = torch.nn.LeakyReLU()
            self.act_f = F.leaky_relu
        elif activation == 'tanh':
            self.act_module = torch.nn.Tanh()
            self.act_f = torch.tanh
        elif activation == 'sigmoid':
            self.act_module = torch.nn.Sigmoid()
            self.act_f = torch.sigmoid

        if self.pooling == 'sum':
            self.pool_f = torch_geometric.nn.global_add_pool
        elif self.pooling == 'mean':
            self.pool_f = torch_geometric.nn.global_mean_pool
        elif self.pooling == 'max':
            self.pool_f = torch_geometric.nn.global_max_pool

        for i in range(self.num_gc_layers):
            if i == 0:
                in_channel, out_channel = n_features, hidden_dim
            elif i == layers-1:
                in_channel, out_channel = hidden_dim, embedding_dim
            else:
                in_channel, out_channel = hidden_dim, hidden_dim

            nn = torch.nn.Sequential(torch.nn.Linear(in_channel, hidden_dim, bias=False),
                                     self.act_module,
                                     torch.nn.Linear(hidden_dim, out_channel, bias=False))
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
