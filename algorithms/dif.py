# -*- coding: utf-8 -*-
# Implementation of Deep Isolation Forest
# @Time    : 2022/8/19
# @Author  : Xu Hongzuo (hongzuo.xu@gmail.com)

import numpy as np
import torch
import random
import time
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pyGDataLoader
from algorithms import net_torch


class DIF:
    """ Class of Deep isolation forest (DIF)
    DIF proposes a new representation scheme, named random representation ensemble,
    creating a group of random representations by optimisation-free neural networks.
    Random axis-parallel cuts are subsequently applied to perform the data partition.

    This representation scheme facilitates high freedom of the partition in the
    original data space (equivalent to non-linear partition on subspaces of varying sizes),
    encouraging a unique synergy between random representations and random partition-based isolation

    Parameters
    ----------
    network_name: str (default='mlp')
        the used network backbone

    network_class: None or torch.nn.Module class (default=None)
        directly feed a network class

    n_ensemble: int (default=50):
        The number of representations in the ensemble

    n_estimators : int (default=6)
        The number of isolation trees per representation.

    max_samples : int or float, default="auto"
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    hidden_dim: list, default=[500,100]
        the list of units number of hidden layers

    rep_dim: int, default=20
        the dimensionality of representations

    skip_connection: None or str, default=None
        whether use skip connection
            - if "concat", then use concatenation-based skip connection

    dropout: None or float, default=None
        whether use dropout in the network
            - if float, represent the dropout probability

    activation: str, default='tanh'
        the name of activation function, {'relu', 'tanh', 'sigmoid', 'leaky_relu'}

    data_type: str, default='tabular'
        the processed data type, {'tabular', 'ts', 'graph'}

    batch_size: int, default=64
        the number of data objects per mini-batch

    random_state : int, RandomState instance or None, default=42
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.
        Pass an int for reproducible results across multiple function calls.

    device: str, default='cuda'
        device for using pytorch, {'cuda', 'cpu'}

    n_processes, int, default=1
        the number of processes during inference

    new_score_func: bool (default=True)
        whether use the proposed new scoring function 
        (Deviation-Enhanced Anomaly Scoring function, DEAS)

    new_ensemble_method: bool (default=True)
        whether use the proposed new ensemble method 
        (Computational-efficient Representation Ensemble, CERE)

    verbose : int, default=0
        Controls the verbosity

    """
    def __init__(self, network_name='mlp', network_class=None,
                 n_ensemble=50, n_estimators=6, max_samples=256,
                 hidden_dim=[500,100], rep_dim=20, skip_connection=None, dropout=None, activation='tanh',
                 data_type='tabular', batch_size=64,
                 new_score_func=True, new_ensemble_method=True,
                 random_state=42, device='cuda', n_processes=1,
                 verbose=0, **network_args):
        # super(DeepIsolationForest, self).__init__(contamination=contamination)

        if data_type not in ['tabular', 'graph', 'ts']:
            raise NotImplementedError('unsupported data type')

        self.data_type = data_type
        self.n_ensemble = n_ensemble
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.batch_size = batch_size

        self.new_score_func = new_score_func
        self.new_ensemble_method = new_ensemble_method

        self.device = device
        self.n_processes = n_processes
        self.verbose = verbose

        self.network_args = network_args
        self.Net = net_torch.choose_net(network_name)
        if network_name == 'mlp':
            self.network_args['n_hidden'] = hidden_dim
            self.network_args['n_emb'] = rep_dim
            self.network_args['skip_connection'] = skip_connection
            self.network_args['dropout'] = dropout
            self.network_args['activation'] = activation
            self.network_args['be_size'] = None if self.new_ensemble_method == False else self.n_ensemble
        elif network_name == 'gin':
            self.network_args['activation'] = activation
        elif network_name == 'dilated_conv':
            self.network_args['hidden_dim'] = hidden_dim
            self.network_args['n_emb'] = rep_dim
        if network_class is not None:
            self.Net = network_class
        print(f'network additional parameters: {network_args}')

        self.transfer_flag = True

        self.n_features = -1
        self.net_lst = []
        self.clf_lst = []
        self.x_reduced_lst = []
        self.score_lst = []

        self.set_seed(random_state)
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
        start_time = time.time()
        self.n_features = X.shape[-1] if self.data_type != 'graph' else max(X.num_features, 1)
        ensemble_seeds = np.random.randint(0, 1e+5, self.n_ensemble)

        if self.verbose >= 2:
            net = self.Net(n_features=self.n_features, **self.network_args)
            print(net)

        self._training_transfer(X, ensemble_seeds)

        if self.verbose >= 2:
            it = tqdm(range(self.n_ensemble), desc='clf fitting', ncols=80)
        else:
            it = range(self.n_ensemble)

        for i in it:
            self.clf_lst.append(
                IsolationForest(n_estimators=self.n_estimators,
                                max_samples=self.max_samples,
                                random_state=ensemble_seeds[i])
            )
            self.clf_lst[i].fit(self.x_reduced_lst[i])

        if self.verbose >= 1:
            print(f'training done, time: {time.time()-start_time:.1f}')
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

        test_reduced_lst = self._inference_transfer(X)
        final_scores = self._inference_scoring(test_reduced_lst, n_processes=self.n_processes)
        return final_scores

    def _training_transfer(self, X, ensemble_seeds):
        if self.new_ensemble_method:
            self.set_seed(ensemble_seeds[0])
            net = self.Net(n_features=self.n_features, **self.network_args).to(self.device)
            self.net_init(net)

            self.x_reduced_lst = self.deep_transfer_batch_ensemble(X, net)
            self.net_lst.append(net)
        else:
            for i in tqdm(range(self.n_ensemble), desc='training ensemble process', ncols=100, leave=None):
                self.set_seed(ensemble_seeds[i])
                net = self.Net(n_features=self.n_features, **self.network_args).to(self.device)
                self.net_init(net)

                self.x_reduced_lst.append(self.deep_transfer(X, net))
                self.net_lst.append(net)
        return

    def _inference_transfer(self, X):
        if self.data_type == 'tabular' and X.shape[0] == self.x_reduced_lst[0].shape[0]:
            return self.x_reduced_lst

        test_reduced_lst = []
        if self.new_ensemble_method:
            test_reduced_lst = self.deep_transfer_batch_ensemble(X, self.net_lst[0])
        else:
            for i in tqdm(range(self.n_ensemble), desc='testing ensemble process', ncols=100, leave=None):
                x_reduced = self.deep_transfer(X, self.net_lst[i])
                test_reduced_lst.append(x_reduced)
        return test_reduced_lst

    def _inference_scoring(self, x_reduced_lst, n_processes):
        if self.new_score_func:
            score_func = self.single_predict
        else:
            score_func = self.single_predict_abla

        n_samples = x_reduced_lst[0].shape[0]
        self.score_lst = np.zeros([self.n_ensemble, n_samples])
        if n_processes == 1:
            for i in range(self.n_ensemble):
                scores = score_func(x_reduced_lst[i], self.clf_lst[i])
                self.score_lst[i] = scores
        else:
            # multiprocessing predict
            start = np.arange(0, self.n_ensemble, np.ceil(self.n_ensemble / n_processes))
            for j in range(int(np.ceil(self.n_ensemble / n_processes))):
                run_id = start + j
                run_id = np.array(np.delete(run_id, np.where(run_id >= self.n_ensemble)), dtype=int)
                if self.verbose >= 1:
                    print('Multi-processing Running ensemble id :', run_id)

                pool = Pool(processes=n_processes)
                process_lst = [pool.apply_async(score_func, args=(x_reduced_lst[i], self.clf_lst[i]))
                               for i in run_id]
                pool.close()
                pool.join()

                for rid, process in zip(run_id, process_lst):
                    self.score_lst[rid] = process.get()

        final_scores = np.average(self.score_lst, axis=0)

        return final_scores


    def deep_transfer(self, X, net):
        x_reduced = []

        with torch.no_grad():
            if self.data_type != 'graph':
                loader = DataLoader(X, batch_size=self.batch_size, drop_last=False, pin_memory=True, shuffle=False)
                for batch_x in loader:
                    batch_x = batch_x.float().to(self.device)
                    batch_x_reduced = net(batch_x)
                    x_reduced.append(batch_x_reduced)
            else:
                loader = pyGDataLoader(X, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False)
                for data in loader:
                    data.to(self.device)
                    x, edge_index, batch = data.x, data.edge_index, data.batch
                    if x is None:
                        x = torch.ones((batch.shape[0], 1)).to(self.device)
                    x, _ = net(x, edge_index, batch)
                    x_reduced.append(x)

        x_reduced = torch.cat(x_reduced).data.cpu().numpy()
        x_reduced = StandardScaler().fit_transform(x_reduced)
        x_reduced = np.tanh(x_reduced)
        return x_reduced

    def deep_transfer_batch_ensemble(self, X, net):
        x_reduced = []

        with torch.no_grad():
            loader = DataLoader(X, batch_size=self.batch_size, drop_last=False, pin_memory=True, shuffle=False)
            for batch_x in loader:
                batch_x = batch_x.float().to(self.device)
                batch_x_reduced = net(batch_x)

                batch_x_reduced = batch_x_reduced.reshape([self.n_ensemble, batch_x.shape[0], -1])
                x_reduced.append(batch_x_reduced)

        x_reduced_lst = [torch.cat([x_reduced[i][j] for i in range(len(x_reduced))]).data.cpu().numpy()
                         for j in range(x_reduced[0].shape[0])]

        for i in range(len(x_reduced_lst)):
            xx = x_reduced_lst[i]
            xx = StandardScaler().fit_transform(xx)
            xx = np.tanh(xx)
            x_reduced_lst[i] = xx

        return x_reduced_lst

    @staticmethod
    def net_init(net):
        for name, param in net.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.normal_(param, mean=0., std=1.)
        return

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def single_predict_abla(x_reduced, clf):
        scores = clf.decision_function(x_reduced)
        scores = -1 * scores
        return scores

    @staticmethod
    def single_predict(x_reduced, clf):
        scores = _cal_score(x_reduced, clf)
        return scores


def _cal_score(xx, clf):
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

        # # slow implementation of deviation calculation
        # t1 = time.time()
        # # calculate deviation in each node of the path
        # # node_deviation_matrix = np.full([xx.shape[0], node_indicator.shape[1]], np.nan)
        # for j in range(xx.shape[0]):
        #     node = np.where(node_indicator[j] == 1)[0]
        #     this_feature_lst = feature_lst[node]
        #     this_threshold_lst = threshold_lst[node]
        #     n_samples_lst = n_node_samples[node]
        #     leaf_samples[j][ii] = n_samples_lst[-1]
        #
        #     deviation = np.abs(xx[j][this_feature_lst[:-1]] - this_threshold_lst[:-1])
        #     if deviation.shape[0] == 0:
        #         print(this_feature_lst[:-1]);print(feature_lst, n_node)
        #
        #     # # directly use mean
        #     deviation = np.mean(deviation)
        #     deviations[j][ii] = deviation
        # print(2, time.time() - t1)

        # # padding node deviation matrix, and use node mean
        # node_deviation_matrix = pd.DataFrame(node_deviation_matrix)
        # for c in node_deviation_matrix.columns:
        #     node_deviation_matrix[c] = node_deviation_matrix[c].fillna(node_deviation_matrix[c].mean())
        #     if pd.isna(node_deviation_matrix[c].mean()):
        #         node_deviation_matrix.drop(c, axis=1, inplace=True)
        #         # node_deviation_matrix[c] = 0
        # node_deviation_matrix = node_deviation_matrix.values
        # deviations[:, ii] = np.mean(node_deviation_matrix, axis=1)

    scores = 2 ** (-depth_sum / (len(clf.estimators_) * _average_path_length([clf.max_samples_])))
    deviation = np.mean(deviations, axis=1)
    leaf_sample = (clf.max_samples_ - np.mean(leaf_samples, axis=1)) / clf.max_samples_

    # print()
    # print('s', scores)
    # print(deviation)
    # print(leaf_sample)

    scores = scores * deviation
    return scores


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

