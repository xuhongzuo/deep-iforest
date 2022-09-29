import os
import numpy as np
import pandas as pd
import re
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from glob import glob


def data_preprocessing(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    x = df.values[:, :-1]
    y = np.array(df.values[:, -1], dtype=int)

    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(x)
    x = minmax_scaler.transform(x)
    return x, y


def min_max_normalize(x):
    filter_lst = []
    for k in range(x.shape[1]):
        s = np.unique(x[:, k])
        if len(s) <= 1:
            filter_lst.append(k)
    if len(filter_lst) > 0:
        print('remove features', filter_lst)
        x = np.delete(x, filter_lst, 1)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    return x


def evaluate(y_true, scores):
    roc_auc = metrics.roc_auc_score(y_true, scores)
    ap = metrics.average_precision_score(y_true, scores)
    return roc_auc, ap


def get_data_lst(dataset_dir, dataset):
    if dataset == 'FULL':
        print(os.path.join(dataset_dir, '*.*'))
        data_lst = glob(os.path.join(dataset_dir, '*.*'))
    else:
        name_lst = dataset.split(',')
        data_lst = []
        for d in name_lst:
            data_lst.extend(glob(os.path.join(dataset_dir, d + '.*')))
    data_lst = sorted(data_lst)
    return data_lst


def adjust_contamination(x, y, contamination_r, swap_ratio=0.05, random_state=42):
    """
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swap 5% features of two anomalies to avoid duplicate contaminated anomalies.
    """
    rng = np.random.RandomState(random_state)

    anom_idx = np.where(y == 1)[0]
    norm_idx = np.where(y == 0)[0]
    n_cur_anom = len(anom_idx)
    n_adj_anom = int(len(norm_idx) * contamination_r / (1. - contamination_r))


    # x_train = np.delete(x_train, unknown_anom_idx, axis=0)
    # y_train = np.delete(y_train, unknown_anom_idx, axis=0)
    # noises = inject_noise(true_anoms, n_adj_noise, 42)
    # x_train = np.append(x_train, noises, axis=0)
    # y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

    # inject noise
    if n_cur_anom < n_adj_anom:
        n_inj_noise = n_adj_anom - n_cur_anom
        print(f'Control Contamination Rate: injecting [{n_inj_noise}] Noisy samples')


        seed_anomalies = x[anom_idx]

        n_sample, dim = seed_anomalies.shape
        n_swap_feat = int(swap_ratio * dim)
        inj_noise = np.empty((n_inj_noise, dim))
        for i in np.arange(n_inj_noise):
            idx = rng.choice(n_sample, 2, replace=False)
            o1 = seed_anomalies[idx[0]]
            o2 = seed_anomalies[idx[1]]
            swap_feats = rng.choice(dim, n_swap_feat, replace=False)
            inj_noise[i] = o1.copy()
            inj_noise[i, swap_feats] = o2[swap_feats]

        x = np.append(x, inj_noise, axis=0)
        y = np.append(y, np.ones(n_inj_noise))

    # remove noise
    elif n_cur_anom > n_adj_anom:
        n_remove = n_cur_anom - n_adj_anom
        print(f'Control Contamination Rate: Removing [{n_remove}] Noise')

        remove_id = anom_idx[rng.choice(n_cur_anom, n_remove, replace=False)]
        print(x.shape)

        x = np.delete(x, remove_id, 0)
        y = np.delete(y, remove_id, 0)
        print(x.shape)

    return x, y



# -------------------------- the following functions are for ts data --------------------------- #

def get_sub_seqs(x_arr, seq_len=100, stride=1, start_discount=np.array([])):
    """
    :param start_discount: the start points of each sub-part in case the x_arr is just multiple parts joined together
    :param x_arr: dim 0 is time, dim 1 is channels
    :param seq_len: size of window used to create subsequences from the data
    :param stride: number of time points the window will move between two subsequences
    :return:
    """
    excluded_starts = []
    [excluded_starts.extend(range((start - seq_len + 1), start)) for start in start_discount if start > seq_len]
    seq_starts = np.delete(np.arange(0, x_arr.shape[0] - seq_len + 1, stride), excluded_starts)
    x_seqs = np.array([x_arr[i:i + seq_len] for i in seq_starts])
    return x_seqs


def get_best_f1(label, score):
    precision, recall, _ = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    return best_f1, best_p, best_r


def get_metrics(label, score):
    auroc = metrics.roc_auc_score(label, score)
    ap = metrics.average_precision_score(y_true=label, y_score=score, average=None)
    best_f1, best_p, best_r = get_best_f1(label, score)

    return auroc, ap, best_f1, best_p, best_r


def get_event_metrics(df, label, score):
    """
    use the corresponding threshold of the best f1 of adjusted scores
    """
    def count_group(*args, df, delta):
        if len(args) == 1:
            df_y = df[df[args[0]] == 1]
        elif len(args) == 2:
            df_y = df[(df[args[0]] == 1) & (df[args[1]] == 1)]
        else:
            raise ValueError("")
        df_y_cur1 = df_y.iloc[:-1, :]
        df_y_cur2 = df_y.iloc[1:, :]
        df_y_cur = [df_y_cur2['time'].iloc[i] - df_y_cur1['time'].iloc[i] for i in range(df_y.shape[0] - 1)]
        num_group = 1
        for i in range(len(df_y_cur)):
            if df_y_cur[i] > pd.Timedelta(delta):
                num_group += 1
        return num_group

    precision, recall, threshold = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_threshold = threshold[np.argmax(f1)]
    label_predict = np.array([s >= best_threshold for s in score], dtype=int)

    # time is previously used as index when reading data frame, reset index to ordered index here
    df = df.reset_index()
    if 'time' in df.columns:
        df_new = df[['time']].copy()
        df_new['time'] = pd.to_datetime(df_new['time']).dt.ceil('S')
        df_new['label'] = label
        df_new['label_predict'] = label_predict

        label_group = count_group('label', df=df_new, delta='12 hour')
        predict_group = count_group('label_predict', df=df_new, delta='12 hour')
        true_group = count_group('label', 'label_predict', df=df_new, delta='12 hour')

        event_precision = true_group / predict_group
        event_recall = true_group / label_group

    else:
        event_precision = -1
        event_recall = -1

    return event_precision, event_recall


def adjust_scores(label, score):
    """
    adjust the score for segment detection. i.e., for each ground-truth anomaly segment,
    use the maximum score as the score of all points in that segment. This corresponds to point-adjust f1-score.
    ** This function is copied/modified from the source code in [Zhihan Li et al. KDD21]
    :param score - anomaly score, higher score indicates higher likelihoods to be anomaly
    :param label - ground-truth label
    """
    score = score.copy()
    assert len(score) == len(label)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
    return score


def get_data_lst_ts(data_root, data, entities=None):
    if type(entities) == str:
        entities_lst = entities.split(',')
    elif type(entities) == list:
        entities_lst = entities
    else:
        raise ValueError('wrong entities')

    name_lst = []
    train_df_lst = []
    test_df_lst = []
    label_lst = []

    if len(glob(os.path.join(data_root, data) + '/*.csv')) == 0:
        machine_lst = os.listdir(data_root + data + '/')
        for m in sorted(machine_lst):
            if entities != 'FULL' and m not in entities_lst:
                continue
            train_path = glob(os.path.join(data_root, data, m, '*train*.csv'))
            test_path = glob(os.path.join(data_root, data, m, '*test*.csv'))

            assert len(train_path) == 1 and len(test_path) == 1, f'{m}'
            train_path, test_path = train_path[0], test_path[0]

            train_df = pd.read_csv(train_path, sep=',', index_col=0)
            test_df = pd.read_csv(test_path, sep=',', index_col=0)
            labels = test_df['label'].values
            train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)

            train_df_lst.append(train_df)
            test_df_lst.append(test_df)
            label_lst.append(labels)
            name_lst.append(m)

        return train_df_lst, test_df_lst, label_lst, name_lst

    else:
        train_df = pd.read_csv(f'{data_root}{data}/{data}_train.csv', sep=',', index_col=0)
        test_df = pd.read_csv(f'{data_root}{data}/{data}_test.csv', sep=',', index_col=0)
        labels = test_df['label'].values
        train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)

        return [train_df], [test_df], [labels], [data]


def eval_ts(scores, labels, test_df):
    eval_info = get_metrics(labels, scores)
    adj_eval_info = get_metrics(labels, adjust_scores(labels, scores))
    event_eval_info = get_event_metrics(test_df, labels, scores)

    eval_info = [round(a, 4) for a in eval_info]
    adj_eval_info = [round(a, 4) for a in adj_eval_info]

    # auroc, ap, best_f1, best_p, best_r, adj_auroc, adj_ap, adj_best_f1, adj_best_p, adj_best_r, event_p, event_r
    # entry = np.concatenate([np.array(eval_info), np.array(adj_eval_info), np.array(event_eval_info)])

    entry = np.array(adj_eval_info)

    return entry


# -------------------------- the following functions are for graph data --------------------------- #

def node_iter(G):
    if float(nx.__version__[:3]) < 2.0:
        return G.nodes()
    else:
        return G.nodes


def node_dict(G):
    if float(nx.__version__[:3]) > 2.1:
        node_dict = G.nodes
    else:
        node_dict = G.node
    return node_dict


def read_graphfile(datadir, dataname, assign_num_node_class=None):
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')
    if assign_num_node_class is not None:
        num_unique_node_labels = assign_num_node_class


    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')

    label_has_zero = False
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []

    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)

    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])

    filename_adj = prefix + '_A.txt'
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))
            index_graph[graph_indic[e0]] += [e0, e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs = []
    for i in range(1, 1 + len(adj_list)):
        G = nx.from_edgelist(adj_list[i])
        G.graph['label'] = graph_labels[i - 1]
        for u in node_iter(G):
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                node_label_one_hot = np.array(node_label_one_hot)
                node_dict(G)[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                node_dict(G)[u]['feat'] = node_attrs[u - 1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        mapping = {}
        it = 0
        for n in node_iter(G):
            mapping[n] = it
            it += 1

        graphs.append(nx.relabel_nodes(G, mapping))
    return graphs

