import os
import pickle
import argparse
import time
import pandas as pd
import numpy as np
import utils
from config import get_algo_config, get_algo_class


dataset_root = 'data'

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=1,
                    help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--input_dir", type=str, default='tabular/', help="the path of the data sets")
parser.add_argument("--output_dir", type=str, default='&tabular_record/',
                    help="the output file path")
parser.add_argument("--dataset", type=str, default='FULL',
                    help="FULL represents all the csv file in the folder, or a list of data set names splitted by comma")
parser.add_argument("--model", type=str, default='dif')
parser.add_argument('--contamination', type=float, default=-1,
                    help='this is used to estimate robustness w.r.t. anomaly contamination')
parser.add_argument('--silent_header', action='store_true')
parser.add_argument('--print_each_run', action='store_true')
parser.add_argument("--note", type=str, default='')


# parameters of DIF
parser.add_argument('--n_ensemble', type=int, default=50)
parser.add_argument('--n_estimators', type=int, default=6)
parser.add_argument('--network_name', type=str, default='layer4-skip')
parser.add_argument('--n_emb', type=int, default=20)


args = parser.parse_args()
if args.model == 'repen':
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF
    import keras.backend as backend
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    KTF.set_session(sess)


os.makedirs(args.output_dir, exist_ok=True)
data_lst = utils.get_data_lst(os.path.join(dataset_root, args.input_dir), args.dataset)
print(data_lst)

model_class = get_algo_class(args.model)
model_configs = get_algo_config(args.model)



if args.model == 'dif':
    model_configs['n_ensemble'] = args.n_ensemble
    model_configs['n_estimators'] = args.n_estimators
    model_configs['network_name'] = args.network_name
    model_configs['data_type'] = 'tabular'
    model_configs['n_emb'] = args.n_emb

print('model configs:', model_configs)

cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())
result_file = os.path.join(args.output_dir, f'{args.model}_results.csv')
raw_res_file = None
if args.print_each_run:
    raw_res_file = os.path.join(args.output_dir, f'{args.model}_{args.input_dir}_{args.contamination}_raw.csv')
    f = open(raw_res_file, 'a')
    print('data,model,auc-roc,auc-pr,time,cont', file=f)

if not args.silent_header:
    f = open(result_file, 'a')
    print('\n---------------------------------------------------------', file=f)
    print(f'model: {args.model}, data dir: {args.input_dir}, dataset: {args.dataset}, contamination: {args.contamination}, {args.runs}runs, ', file=f)
    for k in model_configs.keys():
        print(f'Parameters,\t [{k}], \t\t  {model_configs[k]}', file=f)
    print(f'Note: {args.note}', file=f)
    print('---------------------------------------------------------', file=f)
    print('data, auc-roc, std, auc-pr, std, time', file=f)
    f.close()





for f in data_lst:
    if f.endswith('pkl'):
        df = pd.read_pickle(f)
    elif f.endswith('csv'):
        df = pd.read_csv(f)
    else:
        continue
    dataset_name = os.path.splitext(os.path.split(f)[1])[0]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    x = df.values[:, :-1]
    y = np.array(df.values[:, -1], dtype=int)


    # # Experiment: robustness w.r.t. different anomaly contamination rate
    if args.contamination != -1:
        x_train, y_train = utils.adjust_contamination(x, y,
                                                      contamination_r=args.contamination,
                                                      swap_ratio=0.5,
                                                      random_state=2021)
    else:
        x_train = x
        y_train = y

    auc_lst = np.zeros(args.runs)
    ap_lst = np.zeros(args.runs)
    t_lst = np.zeros(args.runs)
    for i in range(args.runs):
        start_time = time.time()
        print(f'\nRunning [{i+1}/{args.runs}] of [{args.model}] on Dataset [{dataset_name}]')

        if args.model != 'copod':
            clf = model_class(**model_configs, random_state=42 + i)
        else:
            clf = model_class(**model_configs)


        # GOAD uses using normal samples as training data)
        if args.model == 'goad':
            x_train = np.array(x_train, dtype=float)
            x_norm = x_train[np.where(y_train == 0)[0]]
            clf.fit(x_norm)
            scores = clf.decision_function(x)
        else:
            clf.fit(x_train)
            scores = clf.decision_function(x)

        # ablation on representation
        if args.model == 'dif_optim_ae':
            pickle.dump(clf.x_reduced_lst,
                        open(f'&results_emb_euqality/{dataset_name}_{args.model}_reduced_lst_full_anom.pkl', 'wb'))

        # ablation on anomaly scoring
        if args.model == 'dif_knn' or args.model == 'dif_lof' or args.model == 'dif_copod':
            pickle.dump(clf.score_lst, open(f'&results_score_quality/{dataset_name}_{args.model}_score_lst.pkl', 'wb'))


        auc, ap = utils.evaluate(y, scores)
        auc_lst[i], ap_lst[i] = auc, ap
        t_lst[i] = time.time() - start_time

        print('%s, %.4f, %.4f, %.1fs, %s' % (dataset_name, auc_lst[i], ap_lst[i], t_lst[i], args.model))
        if args.print_each_run and raw_res_file is not None:
            txt = f'{dataset_name}, {args.model}, %.4f, %.4f, %.1f, {args.contamination}' % (auc, ap, t_lst[i])
            f = open(raw_res_file, 'a')
            print(txt, file=f)
            f.close()

    avg_auc, avg_ap = np.average(auc_lst), np.average(ap_lst)
    std_auc, std_ap = np.std(auc_lst), np.std(ap_lst)
    avg_time = np.average(t_lst)

    f = open(result_file, 'a')
    txt = f'{dataset_name}, %.4f, %.4f, %.4f, %.4f, %.1f, cont, {args.contamination}' %\
          (avg_auc, std_auc, avg_ap, std_ap, avg_time)
    print(txt, file=f)
    print(txt)
    f.close()


# # done, rename result file
# os.rename(result_file, result_file.replace('.csv', '_done.csv'))


