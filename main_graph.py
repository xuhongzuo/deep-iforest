import os
import argparse
import time
import numpy as np
import torch

import utils
from collections import Counter
from config import get_algo_config, get_algo_class
from pyg_old.pyg_old_tu_dataset import TUDataset


dataset_root = 'data/'

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=1,
                    help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--input_dir", type=str, default='graph/', help="the path of the data sets")
parser.add_argument("--output_dir", type=str, default='&graph_record/',
                    help="the output file path")
parser.add_argument("--dataset", type=str,
                    default='Tox21_MMP',
                    help="FULL represents all the csv file in the folder, or a list of data set names split by comma")
parser.add_argument("--model", type=str, choices=['dif'], default='dif', help="")
parser.add_argument("--note", type=str, default='')

parser.add_argument('--act', type=str, default='tanh')
args = parser.parse_args()

model_class = get_algo_class(args.model)
model_configs = get_algo_config(args.model)


# args.model should be dif only
if args.model == 'dif':
    graph_model_configs = {
        'n_ensemble': 50,
        'data_type': 'graph',
        'network_name': 'gin',
        'new_ensemble_method': 0,
        'batch_size': 5000,
        'n_hidden': 100,
        'n_emb': 50,
        'n_layers': 5,
        'activation': args.act,
        'pooling': 'sum'
    }
    model_configs = dict(model_configs, **graph_model_configs)
    model_configs['network_name'] = 'gin'


# create and print results file header
os.makedirs(args.output_dir, exist_ok=True)
cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())
result_file = os.path.join(args.output_dir, f'graph_{args.model}_{cur_time}.csv')
f = open(result_file, 'a')
print('\n---------------------------------------------------------', file=f)
print(f'model: {args.model}, data dir: {args.input_dir}, dataset: {args.dataset}, {args.runs}runs, ', file=f)
for k in model_configs.keys():
    print(f'Parameters,\t [{k}], \t\t  {model_configs[k]}', file=f)
print(f'Note: {args.note}', file=f)
print(f'---------------------------------------------------------', file=f)
print(f'data, adj_auroc, std, adj_ap, std, adj_f1, std, adj_p, std, adj_r, std, time, model', file=f)
f.close()


os.makedirs(args.output_dir, exist_ok=True)
data_lst = args.dataset.split(',')
print(os.path.join(dataset_root, args.input_dir))
print(data_lst)

data_dir = os.path.join(dataset_root, args.input_dir)
for dataset_name in data_lst:
    path = os.path.join(dataset_root, args.input_dir)
    print('PATH', os.path.join(path, dataset_name+'_training'))
    graphs_train = TUDataset(os.path.join(path, dataset_name + '_training'), name=dataset_name + '_training')
    graphs_test = TUDataset(os.path.join(path, dataset_name + '_testing'), name=dataset_name + '_testing')
    label = graphs_test.data.y
    label = label.flatten().data.numpy()


    # Following PyG, node labels (one-hot vector) are used as node attributes for
    # those datasets that do not provide node attributes
    # But the training and testing data of Tox21 datasets may have different shapes of node labels.
    # padding here to obtain the same length of node attributes
    print(graphs_test.num_features, graphs_train.num_features)
    if graphs_test.num_features != graphs_train.num_features:
        print('padding')
        if graphs_train.num_features > graphs_test.num_features:
            padding_dim = graphs_train.num_features - graphs_test.num_features
            n_test = graphs_test.data.x.shape[0]
            graphs_test.data.x = torch.cat([graphs_test.data.x, torch.zeros(n_test, padding_dim)], dim=1)
        else:
            padding_dim = graphs_test.num_features - graphs_train.num_features
            n_train = graphs_train.data.x.shape[0]
            graphs_train.data.x = torch.cat([graphs_train.data.x, torch.zeros(n_train, padding_dim)], dim=1)


    # use the minority class as anomaly class
    class_key = [a[0] for a in Counter(label).items()]
    class_num = [a[1] for a in Counter(label).items()]
    anom_class = class_key[np.argmin(class_num)]
    y = np.zeros(len(label), dtype=int)
    y[np.where(label == anom_class)[0]] = 1
    print('training num', len(graphs_train))
    print('counter of testing set', Counter(y))

    auc_lst = np.zeros(args.runs)
    ap_lst = np.zeros(args.runs)
    t_lst = np.zeros(args.runs)
    for i in range(args.runs):
        start_time = time.time()
        print(f'\nRunning [{i+1}/{args.runs}] of [{args.model}] on Dataset [{dataset_name}]')

        clf = model_class(**model_configs, random_state=42+i)
        clf.fit(graphs_train)
        scores = clf.decision_function(graphs_test)

        auc, ap = utils.evaluate(y, scores)
        t = round(time.time() - start_time, 1)
        auc_lst[i], ap_lst[i], t_lst[i] = auc, ap, t
        print('%s, %.4f, %.4f, %.1fs, %s' % (dataset_name, auc_lst[i], ap_lst[i], t_lst[i], args.model))

    avg_auc, avg_ap = np.average(auc_lst), np.average(ap_lst)
    std_auc, std_ap = np.std(auc_lst), np.std(ap_lst)
    avg_time = np.average(t_lst)

    f = open(result_file, 'a')
    txt = '%s, %.4f, %.4f, %.4f, %.4f, %.1f' % (dataset_name, avg_auc, std_auc, avg_ap, std_ap, avg_time)
    print(txt, file=f)
    print(txt)
    f.close()


f = open(result_file, 'a')
print('done', file=f)
f.close()
