# -*- coding: utf-8 -*-
# Deep randomised representation with different anomaly scoring methods
# @Time    : 2022/8/19
# @Author  : Xu Hongzuo


import numpy as np
import time
from tqdm import tqdm
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from algorithms.dif import DIF
from multiprocessing import Pool


class DeepRRAD(DIF):
    def __init__(self,
                 network_name='mlp', network_class=None, clf='knn',
                 ensemble_method='batch', n_ensemble=50, n_estimators=6,
                 hidden_dim=[500, 100], rep_dim=20, skip_connection=None, dropout=None, activation='tanh',
                 batch_size=64, device='cuda', n_processes=1, data_type='tabular',
                 post_scal=1, new_score_func=0,
                 verbose=2, random_state=42, **network_args):
        super(DeepRRAD, self).__init__(
            network_name=network_name, network_class=network_class,
            ensemble_method=ensemble_method, n_ensemble=n_ensemble,
            hidden_dim=hidden_dim, rep_dim=rep_dim, skip_connection=skip_connection,
            dropout=dropout, activation=activation,
            batch_size=batch_size, device=device, data_type=data_type,
            new_score_func=0, post_scal=post_scal,
            n_processes=n_processes, verbose=verbose,
            random_state=random_state, **network_args
        )

        self.rng = random_state
        self.set_seed(random_state)

        if clf == 'knn':
            self.Model = KNN
            self.kargs = {'metric': 'euclidean'}
        elif clf == 'lof':
            self.Model = LOF
            self.kargs = {'metric': 'euclidean', 'n_neighbors': 5}
            # self.kargs = {'device': 'cuda:0'}
        elif clf == 'copod':
            self.Model = COPOD
            self.kargs = {}
        elif clf == 'ecod':
            self.Model = ECOD
            self.kargs = {}
        elif clf == 'hbos':
            self.Model = HBOS
            self.kargs = {}
        return

    def fit(self, X, y=None):
        start_time = time.time()
        self.n_features = X.shape[-1] if self.data_type != 'graph' else max(X.num_features, 1)
        ensemble_seeds = np.random.randint(0, 1e+5, self.n_ensemble)

        if self.verbose >= 2:
            net = self.Net(n_features=self.n_features, **self.network_args)
            print(net)

        self.training_transfer(X, ensemble_seeds)

        if self.verbose >= 2:
            it = tqdm(range(self.n_ensemble), desc='clf fitting', ncols=80)
        else:
            it = range(self.n_ensemble)

        for i in it:
            self.clf_lst.append(
                self.Model(**self.kargs)
            )
            self.clf_lst[i].fit(self.x_reduced_lst[i])

        if self.verbose >= 1:
            print(f'training done, time: {time.time()-start_time:.1f}')

        return self


    def inference_scoring(self, x_reduced_lst, n_processes):
        score_func = self.single_predict_abla

        n_samples = x_reduced_lst[0].shape[0]
        self.score_lst = np.zeros([self.n_ensemble, n_samples])
        if n_processes == 1:
            for i in range(self.n_ensemble):
                # transductive learning, using scores in the training stage
                if self.data_type == 'tabular' and \
                        self.x_reduced_lst[0].shape[0] == self.clf_lst[i].decision_scores_.shape[0]:
                    scores = self.clf_lst[i].decision_scores_
                else:
                    scores = score_func(x_reduced_lst[i], self.clf_lst[i])
                # scores = score_func(x_reduced_lst[i], self.clf_lst[i])
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


    @staticmethod
    def single_predict_abla(x_reduced, clf):
        scores = clf.decision_function(x_reduced)
        return scores

