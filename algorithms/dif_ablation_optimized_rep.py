# -*- coding: utf-8 -*-
# @Time    : 2022/8/19
# @Author  : Xu Hongzuo
# @Comment :

from tqdm import tqdm
from algorithms.deep_ad_functions import *
from algorithms.deep_ensemble import DeepEnsembleAD
from sklearn.ensemble import IsolationForest


class DeepORAD(DeepEnsembleAD):
    def __init__(self, n_ensemble=100, ensemble_method='batch', base_model='rdp',
                 epochs=50, batch_size=64, lr=1e-4, epoch_steps=50,
                 hidden_dim=[500,100], rep_dim=20, skip_connection=1, activation='tanh',
                 random_state=42, verbose=2, device='cuda'):
        super(DeepORAD, self).__init__(
            n_ensemble=n_ensemble, ensemble_method=ensemble_method, base_model=base_model,
            epochs=epochs, batch_size=batch_size, lr=lr, epoch_steps=epoch_steps,
            hidden_dim=hidden_dim, rep_dim=rep_dim, skip_connection=skip_connection, activation=activation,
            random_state=random_state, verbose=verbose, device=device
        )
        return

    def decision_function(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                drop_last=False, shuffle=False)
        self.inference(test_loader)

        clf_lst = []
        s_lst = []
        if self.verbose >= 2:
            it = tqdm(range(self.n_ensemble), desc='clf fitting', ncols=80)
        else:
            it = range(self.n_ensemble)

        ensemble_seeds = np.random.randint(0, 1e+5, self.n_ensemble)
        for i in it:
            clf_lst.append(
                IsolationForest(n_estimators=6,
                                max_samples=256,
                                n_jobs=-1,
                                random_state=ensemble_seeds[i])
            )
            clf_lst[i].fit(self.x_reduced_lst[i])
            s_lst.append(-1 * clf_lst[i].decision_function(self.x_reduced_lst[i]))

        final_score = np.average(np.array(s_lst), axis=0)

        return final_score

