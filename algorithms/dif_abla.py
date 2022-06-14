import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from torch.utils.data import DataLoader
from algorithms.dif_pkg import net_torch
from multiprocessing import Pool
from tqdm import tqdm
import random


class DeepRRabla:
    def __init__(self,
                 network_name='layer4', network=None, clf='knn',
                 n_ensemble=50, random_state=42, n_processes=1, verbose=1, post_tanh=True,
                 **network_args):

        self.network_name = network_name

        if network_name == 'layer2':
            self.net = net_torch.Net2
        elif network_name == 'layer3':
            self.net = net_torch.Net3
        elif network_name == 'layer4':
            self.net = net_torch.Net4
        elif network_name == 'layer5':
            self.net = net_torch.Net5
        elif network_name == 'layer6':
            self.net = net_torch.Net6
        elif network_name == 'layer6-skip1':
            self.net = net_torch.Net6S1
        elif network_name == 'layer6-skip2':
            self.net = net_torch.Net6S2
        elif network_name == 'layer6-skip3':
            self.net = net_torch.Net6S3
        elif network_name == 'layer4-skip1':
            self.net = net_torch.Net4S1
        elif network_name == 'layer4-skip2':
            self.net = net_torch.Net4S2
        elif network_name == 'layer4-skip3':
            self.net = net_torch.Net4S3
        elif network_name == 'gru':
            self.net = net_torch.GRUNet
        elif network_name == 'lstm':
            self.net = net_torch.LSTMNet
        elif network_name == 'gin':
            from algorithms.dif_pkg import net_graph
            self.net = net_graph.GinEncoderGraph
        else:
            raise NotImplementedError("")

        if network is not None:
            self.net = network

        self.network_args = network_args

        print(f'network additional parameters: {network_args}')

        self.n_ensemble = n_ensemble
        self.rng = random_state
        self.n_processes = n_processes

        self.post_tanh = post_tanh

        self.net_lst = []
        self.clf_lst = []

        self.x_reduced_lst = []
        self.score_lst = []

        self.decision_scores_ = None

        if self.rng is not None:
            np.random.seed(self.rng)

        self.verbose = verbose

        if clf == 'knn':
            self.Model = KNN
            self.kargs = {'metric': 'euclidean'}
        elif clf == 'lof':
            self.Model = LOF
            self.kargs = {'metric': 'euclidean', 'n_neighbors': 5}
        elif clf == 'copod':
            self.Model = COPOD
            self.kargs = {}
        elif clf == 'hbos':
            self.Model = HBOS
            self.kargs = {}
        return

    def fit(self, X, y=None):

        n_features = X.shape[-1]
        ensemble_seeds = np.random.randint(0, 100000, self.n_ensemble)

        try:
            with tqdm(range(self.n_ensemble)) as pbar:
                pbar.set_description('Deep transfer ensemble:')
                for i in pbar:
                    net = self.net(n_features=n_features, **self.network_args)

                    torch.manual_seed(ensemble_seeds[i])
                    for m in net.modules():
                        if isinstance(m, torch.nn.Linear):
                            torch.nn.init.normal_(m.weight, mean=0., std=1.)

                    x_tensor = torch.from_numpy(X).float()
                    x_reduced = net(x_tensor).data.numpy()

                    if self.post_tanh:
                        # standardize and tanh
                        ss = StandardScaler()
                        x_reduced = ss.fit_transform(x_reduced)
                        x_reduced = np.tanh(x_reduced)


                    self.x_reduced_lst.append(x_reduced)
                    self.net_lst.append(net)
                    self.clf_lst.append(self.Model(**self.kargs))
                    self.clf_lst[i].fit(x_reduced)
        except KeyboardInterrupt:
            pbar.close()
            raise
        pbar.close()

        return self

    def decision_function(self, X):
        x_reduced_lst = []
        try:
            with tqdm(range(self.n_ensemble)) as pbar:
                pbar.set_description('testing deep transfer ensemble:')

                for i in pbar:
                    if X.shape[0] != self.x_reduced_lst[0].shape[0]:
                        # inductive learning
                        x_reduced = self.net_lst[i](torch.from_numpy(X).float()).data.numpy()

                        if self.post_tanh:
                            ss = StandardScaler()
                            x_reduced = ss.fit_transform(x_reduced)
                            x_reduced = np.tanh(x_reduced)
                    else:
                        # transductive learning in tabular data, testing set is identical with training set
                        x_reduced = self.x_reduced_lst[i]


                    x_reduced_lst.append(x_reduced)

        except KeyboardInterrupt:
            pbar.close()
            raise
        pbar.close()


        n_samples = x_reduced_lst[0].shape[0]
        self.score_lst = np.zeros([self.n_ensemble, n_samples])
        if self.n_processes == 1:
            for i in range(self.n_ensemble):
                scores = self.clf_lst[i].decision_function(x_reduced)
                self.score_lst[i] = scores
        else:
            # multiprocessing predict
            start = np.arange(0, self.n_ensemble, np.ceil(self.n_ensemble / self.n_processes))
            for j in range(int(np.ceil(self.n_ensemble / self.n_processes))):
                run_id = start + j
                run_id = np.array(np.delete(run_id, np.where(run_id >= self.n_ensemble)), dtype=int)
                if self.verbose >1:
                    print('Multi-processing Running ensemble id :', run_id)

                pool = Pool(processes=self.n_processes)
                process_lst = [pool.apply_async(single_predict, args=(x_reduced_lst[i], self.clf_lst[i]))
                               for i in run_id]
                pool.close()
                pool.join()

                for rid, process in zip(run_id, process_lst):
                    self.score_lst[rid] = process.get()

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



def single_predict(x_reduced, clf):
    scores = clf.decision_function(x_reduced)
    scores = -1 * scores
    return scores

