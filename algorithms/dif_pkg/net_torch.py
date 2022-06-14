import torch
from torch.nn import functional as F


class Net2(torch.nn.Module):
    def __init__(self, n_features):
        super(Net2, self).__init__()
        self.hidden_layer = torch.nn.Linear(n_features, 20, bias=False)

    def forward(self, x):
        x1 = self.hidden_layer(x)
        return x1


class Net3S3(torch.nn.Module):
    def __init__(self, n_features, n_hidden1=100, n_emb=20, act='tanh'):
        super(Net3S3, self).__init__()
        self.hidden_layer = torch.nn.Linear(n_features, n_hidden1, bias=False)
        self.out_layer = torch.nn.Linear(n_hidden1+n_features, n_emb, bias=False)
        if act == 'tanh':
            self.act_f = torch.tanh
        elif act == 'relu':
            self.act_f = F.relu

    def forward(self, x):
        x1 = self.act_f(self.hidden_layer(x))
        x11 = torch.cat([x1, x], axis=1)
        x2 = self.out_layer(x11)
        return x2


class Net4(torch.nn.Module):
    def __init__(self, n_features, n_hidden1=500, n_hidden2=100, n_hidden3=20, act='tanh'):
        super(Net4, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(n_features, n_hidden1, bias=False)
        self.hidden_layer2 = torch.nn.Linear(n_hidden1, n_hidden2, bias=False)
        self.hidden_layer3 = torch.nn.Linear(n_hidden2, n_hidden3, bias=False)
        if act == 'tanh':
            self.act_f = torch.tanh
        elif act == 'relu':
            self.act_f = F.relu

    def forward(self, x):
        x1 = self.act_f(self.hidden_layer1(x))
        x2 = self.act_f(self.hidden_layer2(x1))
        x3 = self.hidden_layer3(x2)
        return x3


class Net5S3(torch.nn.Module):
    def __init__(self, n_features, n_hidden1=800, n_hidden2=500, n_hidden3=100, n_emb=20, act='tanh'):
        super(Net5S3, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(n_features, n_hidden1, bias=False)
        self.hidden_layer2 = torch.nn.Linear(n_hidden1+n_features, n_hidden2, bias=False)
        self.hidden_layer3 = torch.nn.Linear(n_hidden2+n_hidden1+n_features, n_hidden3, bias=False)
        self.hidden_layer4 = torch.nn.Linear(n_hidden3+n_hidden2+n_hidden1+n_features, n_emb, bias=False)

        if act == 'tanh':
            self.act_f = torch.tanh
        elif act == 'relu':
            self.act_f = F.relu

    def forward(self, x):
        x1 = self.act_f(self.hidden_layer1(x))
        x11 = torch.cat([x1, x], axis=1)

        x2 = self.act_f(self.hidden_layer2(x11))
        x22 = torch.cat([x2, x1, x], axis=1)

        x3 = self.act_f(self.hidden_layer3(x22))
        x33 = torch.cat([x3, x2, x1, x], axis=1)

        x4 = self.hidden_layer4(x33)
        return x4



class Net4S1(torch.nn.Module):
    def __init__(self, n_features, n_hidden1=500, n_hidden2=100, n_hidden3=20, act='tanh'):
        super(Net4S1, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(n_features, n_hidden1, bias=False)
        self.hidden_layer2 = torch.nn.Linear(n_hidden1 + n_features, n_hidden2, bias=False)
        self.hidden_layer3 = torch.nn.Linear(n_hidden2 + n_hidden1, n_hidden3, bias=False)

        if act == 'tanh':
            self.act_f = torch.tanh
        elif act == 'relu':
            self.act_f = F.relu


    def forward(self, x):
        x1 = self.hidden_layer1(x)
        x1 = self.act_f(x1)

        x11 = torch.cat([x1, x], axis=1)
        x2 = self.hidden_layer2(x11)
        x2 = self.act_f(x2)

        x22 = torch.cat([x2, x1], axis=1)
        x3 = self.hidden_layer3(x22)
        return x3


class Net4S2(torch.nn.Module):
    def __init__(self, n_features, n_hidden1=500, n_hidden2=100, n_hidden3=20, act='tanh'):
        super(Net4S2, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(n_features, n_hidden1, bias=False)
        self.hidden_layer2 = torch.nn.Linear(n_hidden1 + n_features, n_hidden2, bias=False)
        self.hidden_layer3 = torch.nn.Linear(n_hidden2 + n_features, n_hidden3, bias=False)

        if act == 'tanh':
            self.act_f = torch.tanh
        elif act == 'relu':
            self.act_f = F.relu


    def forward(self, x):
        x1 = self.hidden_layer1(x)
        x1 = self.act_f(x1)

        x11 = torch.cat([x1, x], axis=1)
        x2 = self.hidden_layer2(x11)
        x2 = self.act_f(x2)

        x22 = torch.cat([x2, x], axis=1)
        x3 = self.hidden_layer3(x22)
        return x3


class Net4S3(torch.nn.Module):
    """
    Dense skip connection
    """
    def __init__(self, n_features, n_hidden1=500, n_hidden2=100, n_emb=20, act='tanh'):
        super(Net4S3, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(n_features, n_hidden1, bias=False)
        self.hidden_layer2 = torch.nn.Linear(n_hidden1 + n_features, n_hidden2, bias=False)
        self.hidden_layer3 = torch.nn.Linear(n_hidden2 + n_hidden1 + n_features, n_emb, bias=False)

        if act == 'tanh':
            self.act_f = torch.tanh
        elif act == 'relu':
            self.act_f = F.relu

    def forward(self, x):
        x1 = self.hidden_layer1(x)
        x1 = self.act_f(x1)

        x11 = torch.cat([x1, x], axis=1)
        x2 = self.hidden_layer2(x11)
        x2 = self.act_f(x2)

        x22 = torch.cat([x2, x1, x], axis=1)
        x3 = self.hidden_layer3(x22)

        # x3 = torch.tanh(x3)
        return x3


class Net6(torch.nn.Module):
    def __init__(self, n_features, n_hidden1=1200, n_hidden2=800, n_hidden3=500, n_hidden4=100, n_hidden5=20, act='tanh'):
        super(Net6, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(n_features, n_hidden1, bias=False)
        self.hidden_layer2 = torch.nn.Linear(n_hidden1, n_hidden2, bias=False)
        self.hidden_layer3 = torch.nn.Linear(n_hidden2, n_hidden3, bias=False)
        self.hidden_layer4 = torch.nn.Linear(n_hidden3, n_hidden4, bias=False)
        self.hidden_layer5 = torch.nn.Linear(n_hidden4, n_hidden5, bias=False)

        if act == 'tanh':
            self.act_f = torch.tanh
        elif act == 'relu':
            self.act_f = F.relu


    def forward(self, x):
        x1 = self.act_f(self.hidden_layer1(x))
        x2 = self.act_f(self.hidden_layer2(x1))
        x3 = self.act_f(self.hidden_layer3(x2))
        x4 = self.act_f(self.hidden_layer4(x3))
        x5 = self.hidden_layer5(x4)
        return x5


class Net6S1(torch.nn.Module):
    def __init__(self, n_features):
        super(Net6S1, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(n_features, 500, bias=False)
        self.hidden_layer2 = torch.nn.Linear(500 + n_features, 400, bias=False)
        self.hidden_layer3 = torch.nn.Linear(400 + 500, 200, bias=False)
        self.hidden_layer4 = torch.nn.Linear(200 + 400, 100, bias=False)
        self.hidden_layer5 = torch.nn.Linear(100 + 200, 20, bias=False)

    def forward(self, x):
        x1 = F.relu(self.hidden_layer1(x))
        x11 = torch.cat([x1, x], axis=1)

        x2 = F.relu(self.hidden_layer2(x11))
        x22 = torch.cat([x2, x1], axis=1)

        x3 = F.relu(self.hidden_layer3(x22))
        x33 = torch.cat([x3, x2], axis=1)

        x4 = F.relu(self.hidden_layer4(x33))
        x44 = torch.cat([x4, x3], axis=1)

        # x5 = F.relu(self.hidden_layer5(x44))
        x5 = self.hidden_layer5(x44)
        return x5


class Net6S2(torch.nn.Module):
    def __init__(self, n_features):
        super(Net6S2, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(n_features, 500, bias=False)
        self.hidden_layer2 = torch.nn.Linear(500 + n_features, 400, bias=False)
        self.hidden_layer3 = torch.nn.Linear(400 + n_features, 200, bias=False)
        self.hidden_layer4 = torch.nn.Linear(200 + n_features, 100, bias=False)
        self.hidden_layer5 = torch.nn.Linear(100 + n_features, 20, bias=False)

    def forward(self, x):
        x1 = F.relu(self.hidden_layer1(x))
        x11 = torch.cat([x1, x], axis=1)

        x2 = F.relu(self.hidden_layer2(x11))
        x22 = torch.cat([x2, x], axis=1)

        x3 = F.relu(self.hidden_layer3(x22))
        x33 = torch.cat([x3, x], axis=1)

        x4 = F.relu(self.hidden_layer4(x33))
        x44 = torch.cat([x4, x], axis=1)

        # x5 = F.relu(self.hidden_layer5(x44))
        x5 = self.hidden_layer5(x44)
        return x5


class Net6S3(torch.nn.Module):
    def __init__(self, n_features, n_hidden1=1000, n_hidden2=800, n_hidden3=500, n_hidden4=100,  n_emb=20):
        super(Net6S3, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(n_features, n_hidden1, bias=False)
        self.hidden_layer2 = torch.nn.Linear(n_hidden1 + n_features, n_hidden2, bias=False)
        self.hidden_layer3 = torch.nn.Linear(n_hidden2 + n_hidden1 + n_features, n_hidden3, bias=False)
        self.hidden_layer4 = torch.nn.Linear(n_hidden3 + n_hidden2 + n_hidden1 + n_features, n_hidden4, bias=False)
        self.hidden_layer5 = torch.nn.Linear(n_hidden4 + n_hidden3 + n_hidden2 + n_hidden1 + n_features, n_emb, bias=False)

    def forward(self, x):
        x1 = torch.tanh(self.hidden_layer1(x))
        x11 = torch.cat([x1, x], axis=1)

        x2 = torch.tanh(self.hidden_layer2(x11))
        x22 = torch.cat([x2, x1, x], axis=1)

        x3 = torch.tanh(self.hidden_layer3(x22))
        x33 = torch.cat([x3, x2, x1, x], axis=1)

        x4 = torch.tanh(self.hidden_layer4(x33))
        x44 = torch.cat([x4, x3, x2, x1, x], axis=1)

        x5 = self.hidden_layer5(x44)
        return x5


class Net7S3(torch.nn.Module):
    def __init__(self, n_features, n_hidden1=1200, n_hidden2=1000, n_hidden3=800, n_hidden4=500, n_hidden5=100,  n_emb=20):
        super(Net7S3, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(n_features, n_hidden1, bias=False)
        self.hidden_layer2 = torch.nn.Linear(n_hidden1 + n_features, n_hidden2, bias=False)
        self.hidden_layer3 = torch.nn.Linear(n_hidden2 + n_hidden1 + n_features, n_hidden3, bias=False)
        self.hidden_layer4 = torch.nn.Linear(n_hidden3 + n_hidden2 + n_hidden1 + n_features, n_hidden4, bias=False)
        self.hidden_layer5 = torch.nn.Linear(n_hidden4 + n_hidden3 + n_hidden2 + n_hidden1 + n_features, n_hidden5, bias=False)
        self.hidden_layer6 = torch.nn.Linear(n_hidden5 + n_hidden4 + n_hidden3 + n_hidden2 + n_hidden1 + n_features, n_emb, bias=False)

    def forward(self, x):
        x1 = torch.tanh(self.hidden_layer1(x))
        x11 = torch.cat([x1, x], axis=1)

        x2 = torch.tanh(self.hidden_layer2(x11))
        x22 = torch.cat([x2, x1, x], axis=1)

        x3 = torch.tanh(self.hidden_layer3(x22))
        x33 = torch.cat([x3, x2, x1, x], axis=1)

        x4 = torch.tanh(self.hidden_layer4(x33))
        x44 = torch.cat([x4, x3, x2, x1, x], axis=1)

        x5 = torch.tanh(self.hidden_layer5(x44))
        x55 = torch.cat([x5, x4, x3, x2, x1, x], axis=1)

        x6 = self.hidden_layer6(x55)
        return x6


class Net8S3(torch.nn.Module):
    def __init__(self, n_features, n_hidden1=1500, n_hidden2=1200, n_hidden3=1000, n_hidden4=800,
                 n_hidden5=500, n_hidden6=100,  n_emb=20):
        super(Net8S3, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(n_features, n_hidden1, bias=False)
        self.hidden_layer2 = torch.nn.Linear(n_hidden1 + n_features, n_hidden2, bias=False)
        self.hidden_layer3 = torch.nn.Linear(n_hidden2 + n_hidden1 + n_features, n_hidden3, bias=False)
        self.hidden_layer4 = torch.nn.Linear(n_hidden3 + n_hidden2 + n_hidden1 + n_features, n_hidden4, bias=False)
        self.hidden_layer5 = torch.nn.Linear(n_hidden4 + n_hidden3 + n_hidden2 + n_hidden1 + n_features,
                                             n_hidden5, bias=False)
        self.hidden_layer6 = torch.nn.Linear(n_hidden5 + n_hidden4 + n_hidden3 + n_hidden2 + n_hidden1 + n_features,
                                             n_hidden6, bias=False)
        self.hidden_layer7 = torch.nn.Linear(n_hidden6 + n_hidden5 + n_hidden4 + n_hidden3 + n_hidden2 + n_hidden1 + n_features,
                                             n_emb, bias=False)

    def forward(self, x):
        x1 = torch.tanh(self.hidden_layer1(x))
        x11 = torch.cat([x1, x], axis=1)

        x2 = torch.tanh(self.hidden_layer2(x11))
        x22 = torch.cat([x2, x1, x], axis=1)

        x3 = torch.tanh(self.hidden_layer3(x22))
        x33 = torch.cat([x3, x2, x1, x], axis=1)

        x4 = torch.tanh(self.hidden_layer4(x33))
        x44 = torch.cat([x4, x3, x2, x1, x], axis=1)

        x5 = torch.tanh(self.hidden_layer5(x44))
        x55 = torch.cat([x5, x4, x3, x2, x1, x], axis=1)

        x6 = torch.tanh(self.hidden_layer6(x55))
        x66 = torch.cat([x6, x5, x4, x3, x2, x1, x], axis=1)

        x7 = self.hidden_layer7(x66)

        return x7



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
