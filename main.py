import os
import pickle
import argparse
import time
import pandas as pd
import numpy as np
import utils
from config import get_algo_config, get_algo_class
from parser_utils import parser_add_model_argument, update_model_configs


dataset_root = 'data'

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=1,
                    help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--input_dir", type=str, default='tabular', help="the path of the data sets")
parser.add_argument("--output_dir", type=str, default='&tabular_record/',
                    help="the output file path")
parser.add_argument("--dataset", type=str, default='FULL',
                    help="FULL represents all the csv file in the folder, or a list of data set names split by comma")
parser.add_argument("--model", type=str, default='dif')
parser.add_argument('--contamination', type=float, default=-1,
                    help='this is used to estimate robustness w.r.t. anomaly contamination')
parser.add_argument('--silent_header', action='store_true')
parser.add_argument('--save_rep', action='store_true')
parser.add_argument('--save_score', action='store_true')
parser.add_argument("--flag", type=str, default='')

parser = parser_add_model_argument(parser)
args = parser.parse_args()


os.makedirs(args.output_dir, exist_ok=True)
data_lst = utils.get_data_lst(os.path.join(dataset_root, args.input_dir), args.dataset)
print(os.path.join(dataset_root, args.input_dir))
print(data_lst)

model_class = get_algo_class(args.model)
model_configs = get_algo_config(args.model)
model_configs = update_model_configs(args, model_configs)
print('model configs:', model_configs)


cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())
result_file = os.path.join(args.output_dir, f'{args.model}.{args.input_dir}.{args.flag}.csv')


if not args.silent_header:
    f = open(result_file, 'a')
    print('\n---------------------------------------------------------', file=f)
    print(f'model: {args.model}, data dir: {args.input_dir}, dataset: {args.dataset}, '
          f'contamination: {args.contamination}, {args.runs}runs, ', file=f)
    for k in model_configs.keys():
        print(f'Parameters,\t [{k}], \t\t  {model_configs[k]}', file=f)
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
    x, y = utils.data_preprocessing(df)

    # # Experiment: robustness w.r.t. different anomaly contamination rate
    if args.contamination != -1:
        x_train, y_train = utils.adjust_contamination(x, y,
                                                      contamination_r=args.contamination,
                                                      swap_ratio=0.5,
                                                      random_state=2021)
    else:
        x_train, y_train = x, y

    auc_lst, ap_lst = np.zeros(args.runs), np.zeros(args.runs),
    t1_lst = np.zeros(args.runs)
    for i in range(args.runs):
        start_time = time.time()
        print(f'\nRunning [{i+1}/{args.runs}] of [{args.model}] on Dataset [{dataset_name}]')

        clf = model_class(**model_configs, random_state=42+i)
        clf.fit(x_train)
        t1 = time.time()
        scores = clf.decision_function(x)

        # # ------ significance of synergy: replacing the random representation ensemble ------ # #
        if args.save_rep and hasattr(clf, "x_reduced_lst"):
            anom_idx, norm_idx = np.where(y == 1)[0], np.where(y == 0)[0]
            if len(norm_idx) > 1000:
                norm_idx = norm_idx[np.random.RandomState(42).choice(len(norm_idx), 1000, replace=False)]

            new_rep_lst = []
            for x_rep in clf.x_reduced_lst:
                anom, norm = x_rep[anom_idx], x_rep[norm_idx]
                x_rep = np.vstack([anom, norm])
                new_rep_lst.append(x_rep)

            save_dir = '&results_rep_quality_2208/'
            os.makedirs(save_dir, exist_ok=True)
            pickle.dump(
                new_rep_lst,
                open(save_dir + f'{dataset_name}_{args.model}_reduced_lst_full_anom.pkl', 'wb')
            )

        # # ------ significance of synergy: replacing the isolation-based anomaly scoring  ------ # #
        if args.save_score and hasattr(clf, "score_lst"):
            save_dir = '&results_score_quality_2208/'
            os.makedirs(save_dir, exist_ok=True)
            pickle.dump(
                clf.score_lst,
                open(save_dir + f'{dataset_name}_{args.model}_score_lst.pkl', 'wb')
            )

        auc, ap = utils.evaluate(y, scores)
        auc_lst[i], ap_lst[i] = auc, ap
        t1_lst[i] = t1 - start_time

        print(f'{dataset_name}, {auc_lst[i]:.4f}, {ap_lst[i]:.4f}, {t1_lst[i]:.1f}, {args.model}')

    avg_auc, avg_ap = np.average(auc_lst), np.average(ap_lst)
    std_auc, std_ap = np.std(auc_lst), np.std(ap_lst)
    avg_time = np.average(t1_lst)

    f = open(result_file, 'a')
    txt = f'{dataset_name}, {avg_auc:.4f}, {std_auc:.4f}, ' \
          f'{avg_ap:.4f}, {std_ap:.4f}, ' \
          f'{avg_time:.1f}, cont, {args.contamination}'
    print(txt, file=f)
    print(txt)
    f.close()
