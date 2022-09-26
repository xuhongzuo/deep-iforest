import numpy as np
import math
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


class DAGMMnet(AEnet, torch.nn.Module):
    def __init__(self, n_features, n_hidden=[500, 100], n_emb=20, est_n_hidden=[10], est_n_gmm=2,
                 activation='tanh',
                 skip_connection=None, be_size=None):
        torch.nn.Module.__init__(self)
        AEnet.__init__(self, n_features=n_features, n_hidden=n_hidden, n_emb=n_emb, activation=activation,
                       skip_connection=skip_connection, be_size=be_size)

        self.est_net = []
        num_layers = len(est_n_hidden)
        for i in range(num_layers+1):
            in_channels = n_emb+2 if i == 0 else est_n_hidden[i-1]
            out_channels = est_n_gmm if i == num_layers else est_n_hidden[num_layers-1-i]
            self.est_net += [LinearBlock(in_channels, out_channels,
                                         bias=False,
                                         activation=activation if i != num_layers else None,
                                         skip_connection=None,
                                         dropout=0.5,
                                         be_size=be_size)]
        self.est_net += [torch.nn.Softmax()]
        self.est_net = torch.nn.Sequential(*self.est_net)

    def forward(self, x):
        if self.be_size is not None:
            x = x.repeat(self.be_size, 1)
        enc = self.encoder(x)
        xx = self.decoder(enc)

        rec1, rec2 = self.compute_reconstruction(x, xx)
        z = torch.cat([enc, rec1.unsqueeze(-1), rec2.unsqueeze(-1)], dim=1)
        gamma = self.est_net(z)
        return enc, xx, z, gamma, x

    @staticmethod
    def compute_reconstruction(x, x_hat):
        relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity


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




# ----------------------------------------- BACKUP ------------------------------------- #

#
# class GraphConv(torch.nn.Module):
#     def __init__(self, input_dim, output_dim, dropout=0.0, add_self=False, normalize_embedding=False, bias=True):
#         super(GraphConv, self).__init__()
#         self.add_self = add_self
#
#         self.dropout = dropout
#         if self.dropout > 0.001:
#             self.dropout_layer = torch.nn.Dropout(p=dropout)
#
#         self.normalize_embedding = normalize_embedding
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.weight = torch.nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
#         if bias:
#             self.bias = torch.nn.Parameter(torch.FloatTensor(output_dim).cuda())
#         else:
#             self.bias = None
#
#     def forward(self, x, adj):
#         if self.dropout > 0.001:
#             x = self.dropout_layer(x)
#         y = torch.matmul(adj, x)
#         if self.add_self:
#             y += x
#         y = torch.matmul(y, self.weight)
#         if self.bias is not None:
#             y = y + self.bias
#         if self.normalize_embedding:
#             y = F.normalize(y, p=2, dim=2)
#         return y

# class GcnEncoderGraph(torch.nn.Module):
#     def __init__(self, n_features, hidden_dim=512, embedding_dim=256, layers=3,
#                  gcn_concat=False, pooling='sum', bn=True, num_aggs=1, dropout=0.0, bias=False):
#         super(GcnEncoderGraph, self).__init__()
#         self.concat = gcn_concat
#         add_self = not gcn_concat
#         self.pooling = pooling
#         self.bn = bn
#         self.num_layers = layers
#         self.num_aggs = num_aggs
#         self.bias = bias
#
#         self.conv_first = GraphConv(input_dim=n_features, output_dim=hidden_dim, add_self=add_self,
#                                     normalize_embedding=True, bias=self.bias)
#         self.conv_block = torch.nn.ModuleList([GraphConv(input_dim=hidden_dim, output_dim=hidden_dim,
#                                                          add_self=add_self, normalize_embedding=True,
#                                                          dropout=dropout, bias=self.bias) for _ in range(layers-2)])
#         self.conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
#                                    normalize_embedding=True, bias=self.bias)
#
#         act = 'relu'
#         if act == 'relu':
#             self.activation = torch.relu
#         elif act == 'leaky_relu':
#             self.activation = F.leaky_relu
#         elif act == 'tanh':
#             self.activation = torch.tanh
#         else:
#             self.activation = torch.relu
#
#
#     def forward(self, x, adj):
#         layers_graph_emb = []
#         layers_node_emb = []
#
#         x = self.conv_first(x, adj)
#         x = self.activation(x)
#         if self.bn:
#             x = self.apply_bn(x)
#         # print('first conv:', x.size()) # shape: [batch_size, n_node, n_emb]
#         layers_node_emb.append(x)
#
#         graph_emb = self.g_pooling(x)
#         layers_graph_emb.append(graph_emb)
#         # print('outall append', out.size()) # shape: [batch_size, n_emb]
#
#
#         for i in range(self.num_layers-2):
#             x = self.conv_block[i](x, adj)
#             x = self.activation(x)
#             if self.bn:
#                 x = self.apply_bn(x)
#             # print('conv', x.size()) # shape: [batch_size, n_node, n_emb]
#
#             # graph_emb = self.g_pooling(x)
#             # layers_graph_emb.append(graph_emb)
#
#         x = self.conv_last(x, adj)
#
#         graph_emb = self.g_pooling(x)
#         layers_graph_emb.append(graph_emb)
#
#         # graph_emb = torch.cat(layers_graph_emb, dim=1)
#
#         # if self.concat:
#         #     graph_emb = torch.cat(layers_graph_emb, dim=1)
#
#         # if self.concat:
#         #     output = torch.cat(out_all, dim=1)
#         # else:
#         #     output = out
#
#         return x, graph_emb
#
#     def g_pooling(self, x):
#         if self.pooling == 'max':
#             graph_emb, _ = torch.max(x, dim=1)
#         elif self.pooling == 'sum':
#             graph_emb = torch.sum(x, dim=1)
#         elif self.pooling == 'max+sum':
#             graph_emb1, _ = torch.max(x, dim=1)
#             graph_emb2 = torch.sum(x, dim=1)
#             graph_emb = torch.cat([graph_emb1, graph_emb2], dim=1)
#         else:
#             raise NotImplementedError('')
#         return graph_emb
#
#     @staticmethod
#     def apply_bn(x):
#         """
#         Batch normalization of 3D tensor x
#         :param x:
#         :return:
#         """
#         bn_module = torch.nn.BatchNorm1d(x.size()[1]).cuda()
#         return bn_module(x)
#
# # GCN#
# # this is referenced from https://github.com/zhulf0804/GCN.PyTorch/
#
# def preprocess_adj(A):
#     '''
#     Pre-process adjacency matrix
#     :param A: adjacency matrix
#     :return:
#     '''
#     I = np.eye(A.shape[0])
#     A_hat = A + I # add self-loops
#     D_hat_diag = np.sum(A_hat, axis=1)
#     D_hat_diag_inv_sqrt = np.power(D_hat_diag, -0.5)
#     D_hat_diag_inv_sqrt[np.isinf(D_hat_diag_inv_sqrt)] = 0.
#     D_hat_inv_sqrt = np.diag(D_hat_diag_inv_sqrt)
#     return np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
#
#
# class GCNLayer(torch.nn.Module):
#     def __init__(self, in_dim, out_dim, acti=True):
#         super(GCNLayer, self).__init__()
#         self.linear = torch.nn.Linear(in_dim, out_dim)
#         if acti:
#             self.acti = torch.nn.ReLU(inplace=True)
#         else:
#             self.acti = None
#     def forward(self, F):
#         output = self.linear(F)
#         if not self.acti:
#             return output
#         return self.acti(output)
#
#
# class GCN(torch.nn.Module):
#     def __init__(self, n_features, hidden_dim, num_classes, p,
#                  gcn_hidden_dim=512, gcn_embedding_dim=256, gcn_layers=3,
#                  gcn_concat=False, gcn_pooling='sum', act='relu', dropout=0.0
#
#                  ):
#         super(GCN, self).__init__()
#         self.gcn_layer1 = GCNLayer(n_features, hidden_dim, acti=True)
#         self.gcn_layer2 = GCNLayer(hidden_dim, num_classes, acti=False)
#         self.dropout = torch.nn.Dropout(dropout)
#
#     def forward(self, X, A):
#         A = torch.from_numpy(preprocess_adj(A)).float()
#         X = self.dropout(X.float())
#         F = torch.mm(A, X)
#
#         F = self.gcn_layer1(F)
#         F = self.dropout(F)
#         F = torch.mm(A, F)
#         output = self.gcn_layer2(F)
#         return output
#
# #
# class AE(torch.nn.Module):
#     def __init__(self, n_features):
#         super(AE, self).__init__()
#
#         # self.hidden_layer1 = torch.nn.Linear(n_features, 500, bias=False)
#         # self.hidden_layer2 = torch.nn.Linear(500, 100, bias=False)
#         # self.hidden_layer3 = torch.nn.Linear(100, 20, bias=False)
#
#         self.encoder = torch.nn.Sequential(
#             torch.nn.Linear(n_features, 500, bias=False),
#             torch.nn.ReLU(),
#             torch.nn.Linear(500, 100, bias=False),
#             torch.nn.ReLU(),
#             torch.nn.Linear(100, 800, bias=False)
#         )
#
#         self.decoder = torch.nn.Sequential(
#             torch.nn.Linear(800, 100, bias=False),
#             torch.nn.ReLU(),
#             torch.nn.Linear(100, 500, bias=False),
#             torch.nn.ReLU(),
#             torch.nn.Linear(500, n_features, bias=False)
#         )
#
#     def forward(self, x):
#         x_enc = self.encoder(x)
#         x_rec = self.decoder(x_enc)
#         return x_enc, x_rec
#
#
#
#
# class FatNet(torch.nn.Module):
#     def __init__(self, n_features):
#         super(FatNet, self).__init__()
#         self.hidden_layer11 = torch.nn.Linear(n_features, 600, bias=False)
#         self.hidden_layer12 = torch.nn.Linear(n_features, 600, bias=False)
#         self.hidden_layer13 = torch.nn.Linear(n_features, 600, bias=False)
#         self.hidden_layer14 = torch.nn.Linear(n_features, 600, bias=False)
#
#         self.hidden_layer21 = torch.nn.Linear(600, 400, bias=False)
#         self.hidden_layer22 = torch.nn.Linear(600, 400, bias=False)
#         self.hidden_layer23 = torch.nn.Linear(600, 400, bias=False)
#         self.hidden_layer24 = torch.nn.Linear(600, 400, bias=False)
#
#         self.hidden_layer3 = torch.nn.Linear(1600, 1200, bias=False)
#
#     def forward(self, x):
#         x11 = F.relu(self.hidden_layer11(x))
#         x12 = F.relu(self.hidden_layer12(x))
#         x13 = F.relu(self.hidden_layer13(x))
#         x14 = F.relu(self.hidden_layer14(x))
#
#         x21 = F.relu(self.hidden_layer21(x11))
#         x22 = F.relu(self.hidden_layer22(x12))
#         x23 = F.relu(self.hidden_layer23(x13))
#         x24 = F.relu(self.hidden_layer24(x14))
#
#         x2 = torch.cat([x21, x22, x23, x24], dim=1)
#
#         x3 = self.hidden_layer3(x2)
#
#         return x3
#
#
# import torch.nn as nn
# class MLP(nn.Module):
#     def __init__(self, n_features,
#                  # n_outputs, n_layers=1, n_units=(1000, 500, 100), nonlinear=nn.Tanh
#                  ):
#         """ The MLP must have the first and last layers as FC.
#         :param n_inputs: input dim
#         :param n_outputs: output dim
#         :param n_layers: layer num = n_layers + 2
#         :param n_units: the dimension of hidden layers
#         :param nonlinear: nonlinear function
#         """
#         super(MLP, self).__init__()
#         self.n_inputs = n_features
#         self.n_outputs = 20
#         self.n_units = (500, 100)
#         self.nonlinear = nn.Tanh
#         self.inv_nonlinear = self.get_inv_nonliner()
#
#         # assert n_layers == len(n_units) - 1
#         self.n_layers = len(self.n_units)-1
#
#         # create layers
#         layers = [nn.Linear(n_features, self.n_units[0], bias=False)]
#         for i in range(self.n_layers):
#             # print(n_units[i], n_units[i+1])
#             layers.append(self.nonlinear())
#             layers.append(nn.Linear(self.n_units[i], self.n_units[i+1], bias=False))
#         layers.append(self.nonlinear())
#         layers.append(nn.Linear(self.n_units[-1], self.n_outputs, bias=False))
#         self.layers = nn.Sequential(*layers)
#
#
#     def get_inv_nonliner(self):
#         """
#         This will return the inverse of the nonlinear function, which is with input as the activation rather than z
#         Currently only support sigmoid and tanh.
#         """
#         if self.nonlinear == nn.Tanh:
#             inv = lambda x: 1 - x * x
#         elif self.nonlinear == nn.Sigmoid:
#             inv = lambda x: x * (1 - x)
#         else:
#             assert False, '{} inverse function is not emplemented'.format(self.nonlinear)
#         return inv
#
#     def forward(self, x):
#
#         x = self.layers(x)
#         return x
#
#     def jacobian(self, x):
#         """
#         :param x: (bs, n_inputs)
#         :return: J (bs, n_outputs, n_inputs)
#         """
#         bs = x.shape[0]
#         # 1. forward pass and get all inverse activation
#         inv_activations = []
#
#         # first do forward
#         for layer_i, layer in enumerate(self.layers):
#             x = layer(x)
#             if layer_i % 2 == 1:  # is activation
#                 inv_activations.append(self.inv_nonlinear(x))
#
#         # 2. compute error in DP fashion
#         len_layers = len(self.layers)
#         len_Deltas = (len_layers + 1) // 2
#         for Delta_i in range(len_Deltas - 1, -1, -1):
#             if Delta_i == len_Deltas - 1:  # if at the final layer, assign it as unit matrix
#                 Delta = torch.diag(torch.ones(self.n_outputs, device=x.device)).unsqueeze(0). \
#                     expand(bs, self.n_outputs, self.n_outputs)
#             else:
#                 layer_i = Delta_i * 2
#                 W = self.layers[layer_i + 2].weight  # current Delta use the previous W
#                 inv_activation_i = Delta_i
#                 inv_activation = inv_activations[inv_activation_i]
#                 Delta = Delta @ (W.unsqueeze(0) * inv_activation.unsqueeze(1))
#
#         # 3. obtain solution with
#         W = self.layers[0].weight
#         J = Delta @ W.unsqueeze(0).expand(bs, self.n_units[0], self.n_inputs)
#         return J
