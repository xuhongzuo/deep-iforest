import numpy as np
import torch
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.utils import check_array
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader



class DeepIsolationForestDSVDDOptim:
    def __init__(self, n_ensemble=50,
                 n_estimators=100, max_samples=256, n_jobs=1,
                 epochs=150, batch_size=128, act='relu',
                 learning_rate=1e-4, weight_decay=0.5e-6,
                 device='cuda', pretrain=True, random_state=42,
                 ):
        # super(DeepIsolationForest, self).__init__(contamination=contamination)

        self.n_ensemble = n_ensemble
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_jobs = n_jobs

        self.latent_dim = 20
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.lr_milestones = [150]
        self.weight_decay = weight_decay
        self.act = act

        self.num_epochs_ae = 150
        self.lr_ae = 1e-4
        self.weight_decay_ae = 0.5e-6

        self.device = device
        self.pretrain = pretrain

        self.rng = random_state
        if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
            torch.backends.cudnn.deterministic=True

        self.n_features = -1
        self.pretrain_dict = {}

        self.net_lst = []
        self.iforest_lst = []
        self.loss_fn = torch.nn.MSELoss()
        self.x_reduced_lst = []

        return

    def set_c(self, model, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        model.eval()
        z_ = []
        with torch.no_grad():
            for x in dataloader:
                x = x.float().to(self.device)
                z = model.encode(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def fit(self, X):
        """Training the Deep SVDD model"""
        self.n_features = X.shape[1]

        ensemble_seeds = np.random.randint(0, 100000, self.n_ensemble)
        for i in range(self.n_ensemble):
            torch.manual_seed(ensemble_seeds[i])
            torch.cuda.manual_seed(ensemble_seeds[i])
            torch.cuda.manual_seed_all(ensemble_seeds[i])
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True, num_workers=0)
            net = MLPNet(X.shape[1], latent_dim=self.latent_dim, act=self.act).to(self.device)
            net = self.train_dsvdd(train_loader, net)


            net.eval()
            with torch.no_grad():
                tensor = torch.from_numpy(X).float().to(self.device)
                Z = net(tensor)
                x_reduced = Z.data.cpu().numpy()

                iforest = IsolationForest(n_estimators=self.n_estimators,
                                          max_samples=self.max_samples,
                                          n_jobs=self.n_jobs,
                                          random_state=self.rng)
                iforest.fit(x_reduced)

                self.x_reduced_lst.append(x_reduced)
                self.iforest_lst.append(iforest)
                self.net_lst.append(net)

        return

    def decision_function(self, X):
        n_sample = X.shape[0]
        final_scores = np.zeros(n_sample)

        for i in range(self.n_ensemble):
            net = self.net_lst[i]

            net.eval()
            with torch.no_grad():
                tensor = torch.from_numpy(X).float().to(self.device)
                x_reduced = net(tensor)
                x_reduced = x_reduced.data.cpu().numpy()

            final_scores += -1 * self.iforest_lst[i].decision_function(x_reduced)

        final_scores = final_scores / self.n_ensemble
        return final_scores


    def train_dsvdd(self, train_loader, net):
        if self.pretrain:
            pretrain_dict = self._pretrain(train_loader)
            net.load_state_dict(pretrain_dict['net_dict'])
            c = torch.from_numpy(pretrain_dict['center']).float().to(self.device)
        else:
            net.apply(weights_init_normal)
            c = torch.randn(self.latent_dim).to(self.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        best_loss = float('inf')
        best_model_dict = None

        net.train()
        try:
            with tqdm(range(self.epochs)) as pbar:
                for _ in pbar:
                    pbar.set_description('Training deep svdd')

                    total_loss = 0
                    for x in train_loader:
                        x = x.float().to(self.device)
                        z = net(x)
                        loss = torch.mean(torch.sum((z - c) ** 2, dim=1))

                        net.zero_grad()
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                    scheduler.step()
                    pbar.set_postfix(loss=total_loss / len(train_loader))

                    if total_loss/len(train_loader) <= best_loss:
                        best_loss = total_loss / len(train_loader)
                        best_model_dict = net.state_dict()

        except KeyboardInterrupt:
            pbar.close()
            raise
        pbar.close()

        # net.load_state_dict(best_model_dict)
        return net


    def _pretrain(self, train_loader):
        """ Pretraining the weights for the deep SVDD network using autoencoder"""
        # ae = AutoEncoder(self.latent_dim).to(self.device)
        ae = MLPAutoEncoder(n_features=self.n_features, act=self.act).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = torch.optim.Adam(ae.parameters(), lr=self.lr_ae, weight_decay=self.weight_decay_ae)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        loss_fn = torch.nn.MSELoss()

        try:
            with tqdm(range(self.epochs)) as pbar:
                ae.train()
                for epoch in pbar:
                    pbar.set_description('Pre-training AE')
                    total_loss = 0
                    for x in train_loader:
                        x = x.float().to(self.device)

                        x_hat = ae(x)
                        # reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                        reconst_loss = loss_fn(x, x_hat)

                        ae.zero_grad()
                        reconst_loss.backward()
                        optimizer.step()

                        total_loss += reconst_loss.item()
                    scheduler.step()

                    pbar.set_postfix(loss=total_loss/len(train_loader))
                    # if (epoch+1) % 10 == 0:
                    #     print('Pretraining Autoencoder... Epoch: {}, Loss: {:.6f}'.
                    #           format(epoch+1, total_loss / len(train_loader)))
        except KeyboardInterrupt:
            pbar.close()
            raise
        pbar.close()

        pretrain_dict = self.save_weights_for_DeepSVDD(ae, train_loader)
        return pretrain_dict

    def save_weights_for_DeepSVDD(self, model, dataloader):
        """Initialize Deep SVDD weights using the encoder weights of the pretrained autoencoder."""
        c = self.set_c(model, dataloader)

        net = MLPNet(self.n_features).to(self.device)

        state_dict = model.state_dict()

        # for a in state_dict.keys():
        #     print(a, state_dict[a].size())
        # print(net)

        net.load_state_dict(state_dict, strict=False)

        pretrain_dict = {'center': c.cpu().data.numpy(), 'net_dict': net.state_dict()}

        # torch.save({'center': c.cpu().data.numpy().tolist(), 'net_dict': net.state_dict()},
        #            'weights/pretrained_parameters.pth')
        return pretrain_dict






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


class MLPNet(torch.nn.Module):
    def __init__(self, n_features, n_hidden1=500, n_hidden2=100, latent_dim=20, act='relu'):
        super(MLPNet, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(n_features, n_hidden1, bias=False)
        self.hidden_layer2 = torch.nn.Linear(n_hidden1, n_hidden2, bias=False)
        self.hidden_layer3 = torch.nn.Linear(n_hidden2, latent_dim, bias=False)
        if act == 'tanh':
            self.act_f = torch.tanh
        elif act == 'relu':
            self.act_f = F.relu

    def forward(self, x):
        x1 = self.act_f(self.hidden_layer1(x))
        x2 = self.act_f(self.hidden_layer2(x1))
        x3 = self.hidden_layer3(x2)
        return x3



class MLPAutoEncoder(torch.nn.Module):
    def __init__(self, n_features, hidden_neurons=[500, 100, 20], act='relu'):
        super(MLPAutoEncoder, self).__init__()
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

    def encode(self, x):
        x = self.act_func(self.hidden_layer1(x))
        x = self.act_func(self.hidden_layer2(x))
        emb = self.act_func(self.hidden_layer3(x))
        return emb


    def decode(self, x):
        x = self.act_func(self.hidden_layer4(x))
        x = self.act_func(self.hidden_layer5(x))
        x = torch.sigmoid(self.hidden_layer6(x))
        return  x

    def forward(self, x):
        emb = self.encode(x)
        x_hat = self.decode(emb)
        return x_hat


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
