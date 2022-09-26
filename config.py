from algorithms import *


def get_algo_class(algo):
    algo_dic = {
        'dif': DIF,
    }
    if algo in algo_dic:
        return algo_dic[algo]

    else:
        raise NotImplementedError("")


def get_algo_config(algo):
    configs = {
        'dif': {
            'ensemble_method': 'batch',
            'n_ensemble': 50,
            'n_estimators': 6,
            'max_samples': 256,
            'batch_size': 64,
            'network_name': 'mlp',
            'data_type': 'tabular',
        },

        'eif': {
            'n_estimators': 300,
            'sample_size': 256,
            'limit': None,
            'ExtensionLevel':'auto'
        },
        'scif': {
            'ndim': 2,
            'sample_size': 256,
            'max_depth': 8,
            'ntrees': 300,
            'ntry': 10,
            'missing_action': 'fail',
            'coefs': 'normal',
            'penalize_range': True,
            'prob_pick_avg_gain': 1
        },
        'pid': {
            'n_estimators': 300,
            'max_samples': 256,
            'max_depth': 8,
        },
        'iforest': {
            'n_estimators': 300,
            'max_samples': 256,
            'n_jobs': -1
        },
        'lesinn': {
            'ensemble_size': 300,
            'subsample_size': 8
        },

        'copod': {

        },
        'repen': {
            'network_depth': 4,
            'epochs': 30,
            'batch_size': 256
        },
        'goad': {
            'n_rots': 64,
            'ndf': 32,
            'd_out': 64,
            'batch_size': 64,
            'n_epoch': 25,
            'lr': 0.001,
            'net_name': 'c5',
        },
        'ae': {
            'epochs': 30,
            'batch_size': 256,
            'hidden_neurons': [500, 100, 20],
            'learning_rate': 1e-3,
            'act': 'tanh'
        },
        'dif_optim_ae':{
            'epochs': 30,
            'batch_size': 256,
            'n_estimators': 10,
            'n_ensemble': 30,
            'learning_rate': 1e-3,
            'act': 'tanh',
        },
        'dif_optim_dsvdd': {
            'epochs': 30,
            'batch_size': 256,
            'n_ensemble': 30,
            'n_estimators': 10,
            'learning_rate': 1e-3,
            'act': 'tanh',
            'pretrain': False
        },
        'dif_copod': {
            'n_ensemble': 50,
            'n_processes': 1,
            'network_name': 'layer4-skip3',
            'clf': 'copod'
        },
        'dif_knn': {
            'n_ensemble': 50,
            'n_processes': 1,
            'network_name': 'layer4-skip3',
            'clf': 'knn'
        },
        'dif_lof': {
            'n_ensemble': 50,
            'n_processes': 1,
            'network_name': 'layer4-skip3',
            'clf': 'lof'
        },
    }
    assert algo in list(configs.keys()), f"Algo name {algo} not identified"
    return configs[algo]