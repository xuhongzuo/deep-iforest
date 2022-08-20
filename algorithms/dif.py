import numpy as np
import networkx as nx
import torch
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader
from algorithms.dif_pkg import net_util
from tqdm import tqdm


class DeepIsolationForest:
    def __init__(self, network_name='layer4', network=None, n_ensemble=50, n_estimators=6, max_samples=256,
                 n_jobs=1, random_state=42, n_processes=15, data_type='tabular',
                 batch_size=10000, device='cuda', graph_feature_type='default',
                 verbose=2,
                 **network_args):

        self.network_name = network_name
        self.net = net_util.choose_net(network_name)
        if network is not None:
            self.net = network
        self.network_args = network_args

        if network_name.startswith('layer'):
            n_layer = int(network_name[5]) - 2
            is_skip = 1 if network_name.endswith('skip') else 0
            hidden_size = [1200, 800, 500, 100]
            self.network_args['n_hidden'] = hidden_size[-n_layer::]
            self.network_args['skip_connection'] = is_skip

        print(f'network additional parameters: {network_args}')

        self.n_ensemble = n_ensemble
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.rng = random_state
        self.n_processes = n_processes


        self.batch_size = batch_size
        self.device = device
        self.graph_feature_type = graph_feature_type

        self.data_type = data_type

        self.net_lst = []
        self.iForest_lst = []
        self.x_reduced_lst = []
        self.score_lst = []

        self.decision_scores_ = None

        if self.rng is not None:
            np.random.seed(self.rng)

        self.verbose = verbose

        return

    def fit(self, X, y=None):
        """
        Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """

        if self.data_type == 'tabular' or self.data_type == 'ts':
            # tabular data are in 2-dim data, n_features should be X.shape[1]
            # ts data are in 3-dim data, n_features should be in X.shape[2]
            n_features = X.shape[-1]
        elif self.data_type == 'graph':
            n_features = max(X.num_features, 1)
        else:
            raise NotImplementedError('')

        ensemble_seeds = np.random.randint(0, 100000, self.n_ensemble)

        if self.verbose >= 2:
            net = self.net(n_features=n_features, **self.network_args)
            print(net)

        # for each ensemble
        try:
            with tqdm(range(self.n_ensemble)) as pbar:
                pbar.set_description('Deep transfer ensemble:')
                for i in pbar:
                    # -------------------------------- for tabular data -------------------------- #
                    if self.data_type == 'tabular':
                        net = self.net(n_features=n_features, **self.network_args)

                        torch.manual_seed(ensemble_seeds[i])
                        for m in net.modules():
                            if isinstance(m, torch.nn.Linear):
                                torch.nn.init.normal_(m.weight, mean=0., std=1.)

                        x_tensor = torch.from_numpy(X).float()
                        x_reduced = net(x_tensor).data.numpy()

                        ss = StandardScaler()
                        x_reduced = ss.fit_transform(x_reduced)
                        x_reduced = np.tanh(x_reduced)

                    # -------------------------------- for ts data -------------------------- #
                    elif self.data_type == 'ts':
                        net = self.net(n_features=n_features, **self.network_args)
                        net = net.to(self.device)

                        torch.manual_seed(ensemble_seeds[i])
                        for name, param in net.named_parameters():
                            torch.nn.init.normal_(param, mean=0., std=1.)
                        x_reduced = self.deep_transfer(X, net, self.batch_size, self.device)

                    # -------------------------------- for graph data -------------------------- #
                    elif self.data_type == 'graph':
                        net = self.net(n_features=n_features, **self.network_args)
                        net = net.to(self.device)

                        torch.manual_seed(ensemble_seeds[i])
                        for m in net.modules():
                            if isinstance(m, torch.nn.Linear):
                                torch.nn.init.normal_(m.weight.data, mean=0., std=1.)

                        x_reduced = self.graph_deep_transfer(X, net, self.batch_size, self.device)

                    else:
                        raise NotImplementedError('')


                    self.x_reduced_lst.append(x_reduced)
                    self.net_lst.append(net)
                    self.iForest_lst.append(IsolationForest(n_estimators=self.n_estimators,
                                                            max_samples=self.max_samples,
                                                            n_jobs=self.n_jobs,
                                                            random_state=ensemble_seeds[i]))
                    self.iForest_lst[i].fit(x_reduced)
        except KeyboardInterrupt:
            pbar.close()
            raise
        pbar.close()

        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """

        x_reduced_lst = []
        try:
            with tqdm(range(self.n_ensemble)) as pbar:
                pbar.set_description('testing deep transfer ensemble:')

                for i in pbar:
                    if self.data_type == 'tabular':
                        if X.shape[0] != self.x_reduced_lst[0].shape[0]:
                            x_reduced = self.net_lst[i](torch.from_numpy(X).float()).data.numpy()
                            ss = StandardScaler()
                            x_reduced = ss.fit_transform(x_reduced)
                            x_reduced = np.tanh(x_reduced)
                        else:
                            # transductive learning in tabular data, testing set is identical with training set
                            x_reduced = self.x_reduced_lst[i]

                    elif self.data_type == 'ts':
                        x_reduced = self.deep_transfer(X, self.net_lst[i], self.batch_size, self.device)

                    elif self.data_type == 'graph':
                        x_reduced = self.graph_deep_transfer(X, self.net_lst[i], self.batch_size, self.device)

                    else:
                        raise NotImplementedError('')
                    x_reduced_lst.append(x_reduced)

        except KeyboardInterrupt:
            pbar.close()
            raise
        pbar.close()


        n_samples = x_reduced_lst[0].shape[0]
        self.score_lst = np.zeros([self.n_ensemble, n_samples])
        for i in range(self.n_ensemble):
            scores = single_predict(x_reduced_lst[i], self.iForest_lst[i])
            self.score_lst[i] = scores
        final_scores = np.average(self.score_lst, axis=0)


        return final_scores

    @staticmethod
    def deep_transfer(X, net, batch_size, device):
        x_reduced = []
        loader = DataLoader(dataset=X, batch_size=batch_size, drop_last=False, pin_memory=True, shuffle=False)
        for batch_x in loader:
            batch_x = batch_x.float().to(device)
            batch_x_reduced = net(batch_x).data.cpu().numpy()
            x_reduced.extend(batch_x_reduced)
        x_reduced = np.array(x_reduced)
        return x_reduced

    @staticmethod
    def graph_deep_transfer(X, net, batch_size, device):
        from torch_geometric.data import DataLoader as pyGDataLoader
        x_reduced = []
        loader = pyGDataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
        for data in loader:
            data.to(device)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            if x is None:
                x = torch.ones((batch.shape[0], 1)).to(device)
            x, _ = net(x, edge_index, batch)
            x_reduced.extend(x.data.cpu().numpy())

        x_reduced = np.array(x_reduced)
        return x_reduced


def single_predict(x_reduced, clf):
    scores = cal_score(x_reduced, clf)
    return scores


def cal_score(xx, clf):
    depths = np.zeros((xx.shape[0], len(clf.estimators_)))
    depth_sum = np.zeros(xx.shape[0])
    deviations = np.zeros((xx.shape[0], len(clf.estimators_)))
    leaf_samples = np.zeros((xx.shape[0], len(clf.estimators_)))

    for ii, estimator_tree in enumerate(clf.estimators_):
        # estimator_population_ind = sample_without_replacement(n_population=xx.shape[0], n_samples=256,
        #                                                       random_state=estimator_tree.random_state)
        # estimator_population = xx[estimator_population_ind]

        tree = estimator_tree.tree_
        n_node = tree.node_count

        if n_node == 1:
            continue

        # get feature and threshold of each node in the iTree
        # in feature_lst, -2 indicates the leaf node
        feature_lst, threshold_lst = tree.feature.copy(), tree.threshold.copy()

        #     feature_lst = np.zeros(n_node, dtype=int)
        #     threshold_lst = np.zeros(n_node)
        #     for j in range(n_node):
        #         feature, threshold = tree.feature[j], tree.threshold[j]
        #         feature_lst[j] = feature
        #         threshold_lst[j] = threshold
        #         # print(j, feature, threshold)
        #         if tree.children_left[j] == -1:
        #             leaf_node_list.append(j)

        # compute depth and score
        leaves_index = estimator_tree.apply(xx)
        node_indicator = estimator_tree.decision_path(xx)

        # The number of training samples in each test sample leaf
        n_node_samples = estimator_tree.tree_.n_node_samples

        # node_indicator is a sparse matrix with shape (n_samples, n_nodes), indicating the path of input data samples
        # each layer would result in a non-zero element in this matrix,
        # and then the row-wise summation is the depth of data sample
        n_samples_leaf = estimator_tree.tree_.n_node_samples[leaves_index]
        d = (np.ravel(node_indicator.sum(axis=1)) + _average_path_length(n_samples_leaf) - 1.0)
        depths[:, ii] = d
        depth_sum += d

        # decision path of data matrix XX
        node_indicator = np.array(node_indicator.todense())

        # set a matrix with shape [n_sample, n_node], representing the feature value of each sample on each node
        # set the leaf node as -2
        value_mat = np.array([xx[i][feature_lst] for i in range(xx.shape[0])])
        value_mat[:, np.where(feature_lst == -2)[0]] = -2
        th_mat = np.array([threshold_lst for _ in range(xx.shape[0])])

        mat = np.abs(value_mat - th_mat) * node_indicator

        # dev_mat = np.abs(value_mat - th_mat)
        # m = np.mean(dev_mat, axis=0)
        # s = np.std(dev_mat, axis=0)
        # dev_mat_mean = np.array([m for _ in range(xx.shape[0])])
        # dev_mat_std = np.array([s for _ in range(xx.shape[0])])
        # dev_mat_zscore = np.maximum((dev_mat - dev_mat_mean) / (dev_mat_std+1e-6), 0)
        # mat = dev_mat_zscore * node_indicator

        exist = (mat != 0)
        dev = mat.sum(axis=1)/(exist.sum(axis=1)+1e-6)
        deviations[:, ii] = dev

    scores = 2 ** (-depth_sum / (len(clf.estimators_) * _average_path_length([clf.max_samples_])))
    deviation = np.mean(deviations, axis=1)
    leaf_sample = (clf.max_samples_ - np.mean(leaf_samples, axis=1)) / clf.max_samples_

    scores = scores * deviation
    # scores = scores * deviation * leaf_sample
    return scores



def get_depth(x_reduced, clf):
    n_samples = x_reduced.shape[0]

    depths = np.zeros((n_samples, len(clf.estimators_)))
    depth_sum = np.zeros(n_samples)
    for ii, (tree, features) in enumerate(zip(clf.estimators_, clf.estimators_features_)):
        leaves_index = tree.apply(x_reduced)
        node_indicator = tree.decision_path(x_reduced)
        n_samples_leaf = tree.tree_.n_node_samples[leaves_index]

        # node_indicator is a sparse matrix, indicating the path of input data samples
        # with shape (n_samples, n_nodes)
        # each layer would result in a non-zero element in this matrix,
        # and then the row-wise summation is the depth of data sample
        d = (np.ravel(node_indicator.sum(axis=1)) + _average_path_length(n_samples_leaf) - 1.0)
        depths[:, ii] = d
        depth_sum += d

    scores = 2 ** (-depth_sum / (len(clf.estimators_) * _average_path_length([clf.max_samples_])))
    return depths, scores


def _average_path_length(n_samples_leaf):
    """
    The average path length in a n_samples iTree, which is equal to
    the average path length of an unsuccessful BST search since the
    latter has the same structure as an isolation tree.
    Parameters
    ----------
    n_samples_leaf : array-like of shape (n_samples,)
        The number of training samples in each test sample leaf, for
        each estimators.

    Returns
    -------
    average_path_length : ndarray of shape (n_samples,)
    """

    n_samples_leaf = check_array(n_samples_leaf, ensure_2d=False)

    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.
    average_path_length[mask_2] = 1.
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)


class GraphData(torch.utils.data.Dataset):
    """Sample graphs and nodes in graph"""

    def __init__(self, G_list, features='default', normalize=True, max_num_nodes=0):
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []

        self.assign_feat_all = []
        self.max_num_nodes = max_num_nodes

        if features == 'default':
            self.feat_dim = node_dict(G_list[0])[0]['feat'].shape[0]
        elif features == 'node_label':
            self.feat_dim = node_dict(G_list[0])[0]['label'].shape[0]

        for G in G_list:
            adj = np.array(nx.to_numpy_matrix(G))
            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            self.adj_all.append(adj)
            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph['label'])
            # attributed graph?
            if features == 'default':
                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i, u in enumerate(G.nodes()):
                    f[i, :] = node_dict(G)[u]['feat']
                self.feature_all.append(f)
            # for graph that does not have node features
            elif features == 'deg-num':
                degs = np.sum(np.array(adj), 1)
                if self.max_num_nodes > G.number_of_nodes():
                    degs = np.expand_dims(np.pad(degs, (0, self.max_num_nodes - G.number_of_nodes()),
                                                 'constant', constant_values=0), axis=1)
                elif self.max_num_nodes < G.number_of_nodes():
                    deg_index = np.argsort(degs, axis=0)
                    deg_ind = deg_index[0: G.number_of_nodes() - self.max_num_nodes]
                    degs = np.delete(degs, [deg_ind], axis=0)
                    degs = np.expand_dims(degs, axis=1)
                else:
                    degs = np.expand_dims(degs, axis=1)
                self.feature_all.append(degs)
            elif features == 'node_label':
                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i, u in enumerate(G.nodes()):
                    f[i, :] = node_dict(G)[u]['label']
                self.feature_all.append(f)



            self.assign_feat_all.append(self.feature_all[-1])

        self.feat_dim = self.feature_all[0].shape[1]
        self.assign_feat_dim = self.assign_feat_all[0].shape[1]

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        if self.max_num_nodes > num_nodes:
            adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
            adj_padded[:num_nodes, :num_nodes] = adj
        elif self.max_num_nodes < num_nodes:
            degs = np.sum(np.array(adj), 1)
            deg_index = np.argsort(degs, axis=0)
            deg_ind = deg_index[0:num_nodes - self.max_num_nodes]
            adj_padded = np.delete(adj, [deg_ind], axis=0)
            adj_padded = np.delete(adj_padded, [deg_ind], axis=1)
        else:
            adj_padded = adj

        return {'adj': adj_padded,
                'feats': self.feature_all[idx].copy(),
                'label': self.label_all[idx],
                # 'num_nodes': num_nodes,
                # 'assign_feats':self.assign_feat_all[idx].copy()
                }


def node_dict(G):
    if float(nx.__version__[:3]) > 2.1:
        dict_ = G.nodes
    else:
        dict_ = G.node
    return dict_
