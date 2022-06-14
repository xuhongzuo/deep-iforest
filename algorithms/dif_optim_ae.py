import numpy as np
import torch
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader



class AENetEmb(torch.nn.Module):
    def __init__(self, n_features, hidden_neurons=[500, 100, 20], act='relu'):
        super(AENetEmb, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(n_features, hidden_neurons[0], bias=False)
        self.hidden_layer2 = torch.nn.Linear(hidden_neurons[0], hidden_neurons[1], bias=False)
        self.hidden_layer3 = torch.nn.Linear(hidden_neurons[1], hidden_neurons[2], bias=False)
        self.hidden_layer4 = torch.nn.Linear(hidden_neurons[2], hidden_neurons[1], bias=False)
        self.hidden_layer5 = torch.nn.Linear(hidden_neurons[1], hidden_neurons[0], bias=False)
        self.hidden_layer6 = torch.nn.Linear(hidden_neurons[0], n_features, bias=False)

        if act == 'relu':
            self.act_func = torch.relu
        elif act == 'tanh':
            self.act_func = torch.tanh


    def forward(self, x):
        x = self.act_func(self.hidden_layer1(x))
        x = self.act_func(self.hidden_layer2(x))
        emb = self.act_func(self.hidden_layer3(x))
        x = self.act_func(self.hidden_layer4(emb))
        x = self.act_func(self.hidden_layer5(x))
        x = torch.sigmoid(self.hidden_layer6(x))
        return emb, x


class AENetEmb2(torch.nn.Module):
    def __init__(self, n_features, hidden_neurons=[100, 20], act='relu'):
        super(AENetEmb2, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(n_features, hidden_neurons[0], bias=False)
        self.hidden_layer2 = torch.nn.Linear(hidden_neurons[0], hidden_neurons[1], bias=False)
        self.hidden_layer3 = torch.nn.Linear(hidden_neurons[1], hidden_neurons[0], bias=False)
        self.hidden_layer4 = torch.nn.Linear(hidden_neurons[0], n_features, bias=False)

        if act == 'relu':
            self.act_func = torch.relu
        elif act == 'tanh':
            self.act_func = torch.tanh


    def forward(self, x):
        x = self.act_func(self.hidden_layer1(x))
        emb = self.act_func(self.hidden_layer2(x))
        x = self.act_func(self.hidden_layer3(emb))
        x = torch.sigmoid(self.hidden_layer4(x))
        return emb, x


class DeepIsolationForestAEOptim:
    def __init__(self,
                 n_ensemble=50, n_estimators=6, max_samples=256,
                 n_jobs=1, random_state=42, act = 'relu', learning_rate=1e-4,
                 epochs=150, batch_size=128, device='cuda',
                 verbose=1,
                 **network_args):

        self.network_args = network_args
        print(f'network additional parameters: {network_args}')

        self.n_ensemble = n_ensemble

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.rng = random_state

        self.act = act

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate

        self.device = device

        self.decision_scores_ = None

        if self.rng is not None:
            np.random.seed(self.rng)

        self.verbose = verbose

        self.net_lst = []
        self.iforest_lst = []
        self.loss_fn = torch.nn.MSELoss()
        self.x_reduced_lst = []
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

        n_samples, n_features = X.shape[0], X.shape[1]

        ensemble_seeds = np.random.randint(0, 100000, self.n_ensemble)

        for i in range(self.n_ensemble):
            torch.manual_seed(ensemble_seeds[i])
            torch.cuda.manual_seed(ensemble_seeds[i])
            torch.cuda.manual_seed_all(ensemble_seeds[i])
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            train_loader = torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=True)

            net = AENetEmb2(n_features=n_features, hidden_neurons=[100,20], act=self.act)
            # net = AENetEmb(n_features=n_features, hidden_neurons=[500, 100, 20], act=self.act)
            net = self._train_autoencoder(train_loader, net)


            net.eval()
            with torch.no_grad():
                tensor = torch.from_numpy(X).float().to(self.device)
                emb, _ = net(tensor)
                x_reduced = emb.data.cpu().numpy()

            iforest = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                      n_jobs=self.n_jobs, random_state=ensemble_seeds[i])
            iforest.fit(x_reduced)

            self.x_reduced_lst.append(x_reduced)
            self.iforest_lst.append(iforest)
            self.net_lst.append(net)
        return self

    def decision_function(self, X):
        n_sample = X.shape[0]
        final_scores = np.zeros(n_sample)
        for i in range(self.n_ensemble):
            tensor = torch.from_numpy(X).float().to(self.device)

            self.net_lst[i].eval()
            with torch.no_grad():
                emb, _ = self.net_lst[i](tensor)
                emb = emb.data.cpu().numpy()

            final_scores += -1 * self.iforest_lst[i].decision_function(emb)
        final_scores = final_scores / self.n_ensemble

        return final_scores

    def _train_autoencoder(self, train_loader, net):
        """Internal function to train the autoencoder

        Parameters
        ----------
        train_loader : torch dataloader
            Train data.
        """
        net = net.to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-5)

        best_loss = float('inf')
        best_model_dict = None


        net.train()
        try:
            with tqdm(range(self.epochs)) as pbar:
                for epoch in pbar:
                    pbar.set_description('Training AE: ')
                    overall_loss = []
                    for data in train_loader:
                        data = data.to(self.device).float()
                        emb, rec_data = net(data)
                        loss = self.loss_fn(data, rec_data)

                        net.zero_grad()
                        loss.backward()
                        optimizer.step()
                        overall_loss.append(loss.item())

                    if self.verbose >= 2:
                        if (epoch+1) % 10 == 0:
                            print('epoch {epoch}: training loss {train_loss} '.format(
                                epoch=epoch+1, train_loss=np.mean(overall_loss)))
                    elif self.verbose == 1:
                        pbar.set_postfix(loss=np.mean(overall_loss))

                    # track the best model so far
                    if np.mean(overall_loss) <= best_loss:
                        best_loss = np.mean(overall_loss)
                        best_model_dict = net.state_dict()

        except KeyboardInterrupt:
            pbar.close()
            raise
        pbar.close()

        # net.load_state_dict(best_model_dict)
        return net




