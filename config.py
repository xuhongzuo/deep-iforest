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
            'n_ensemble': 50,
            'n_estimators': 6,
            'max_samples': 256,
            'batch_size': 64,
            'network_name': 'mlp',
            'data_type': 'tabular',
        },
    }
    assert algo in list(configs.keys()), f"Algo name {algo} not identified"
    return configs[algo]