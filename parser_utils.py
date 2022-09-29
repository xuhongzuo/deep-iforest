

def parser_add_model_argument(parser):
    """add parameters of DIF into parser"""
    parser.add_argument('--n_ensemble', type=int, default=50)

    # parameters of DIF
    parser.add_argument('--n_estimators', type=int, default=6)
    parser.add_argument('--rep_dim', type=int, default=20)
    parser.add_argument('--hidden_dim', type=str, default='500,100')
    parser.add_argument('--skip_c', type=int, default=1)
    parser.add_argument('--act', type=str, default='tanh')
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--new_score_func', type=int, default=1)
    parser.add_argument('--new_ensemble_method', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')

    return parser

def update_model_configs(args, model_configs):
    """update model configs by args"""
    model_configs['n_ensemble'] = args.n_ensemble
    model_configs['rep_dim'] = args.rep_dim
    model_configs['hidden_dim'] = args.hidden_dim
    model_configs['skip_connection'] = 'concat' if args.skip_c == 1 else None
    model_configs['activation'] = args.act

    model_configs['new_score_func'] = args.new_score_func
    model_configs['new_ensemble_method'] = args.new_ensemble_method

    model_configs['n_estimators'] = args.n_estimators
    model_configs['n_processes'] = args.n_processes
    model_configs['network_name'] = 'mlp'
    model_configs['data_type'] = 'tabular'
    model_configs['batch_size'] = args.batch_size
    model_configs['device'] = args.device

    return model_configs