

def parser_add_model_argument(parser):
    parser.add_argument('--n_ensemble', type=int, default=50)

    # parameters of DIF
    parser.add_argument('--n_estimators', type=int, default=6)
    parser.add_argument('--rep_dim', type=int, default=20)
    parser.add_argument('--hidden_dim', type=str, default='500,100')
    parser.add_argument('--skip_c', type=int, default=1)
    parser.add_argument('--act', type=str, default='tanh')
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--post_scal', type=int, default=1)
    parser.add_argument('--new_score_func', type=int, default=1)
    parser.add_argument('--ensemble_method', type=str, default='batch')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')

    return parser

def update_model_configs(args, model_configs):
    if args.model == 'dif' or args.model.startswith('deeprr'):
        model_configs['n_ensemble'] = args.n_ensemble
        model_configs['rep_dim'] = args.rep_dim
        model_configs['hidden_dim'] = args.hidden_dim
        model_configs['skip_connection'] = 'concat' if args.skip_c == 1 else None
        model_configs['activation'] = args.act

        model_configs['post_scal'] = args.post_scal
        model_configs['new_score_func'] = args.new_score_func

        model_configs['n_estimators'] = args.n_estimators
        model_configs['n_processes'] = args.n_processes
        model_configs['network_name'] = 'mlp'
        model_configs['data_type'] = 'tabular'
        model_configs['ensemble_method'] = args.ensemble_method
        model_configs['batch_size'] = args.batch_size
        model_configs['device'] = args.device

    elif args.model.startswith('deepor'):
        model_configs['n_ensemble'] = args.n_ensemble
        model_configs['rep_dim'] = args.rep_dim
        model_configs['hidden_dim'] = args.hidden_dim
        model_configs['skip_connection'] = 'concat' if args.skip_c == 1 else None
        model_configs['activation'] = args.act

        model_configs['epochs'] = args.epochs
        model_configs['epoch_steps'] = args.epoch_steps
        model_configs['batch_size'] = args.batch_size
        model_configs['lr'] = args.lr
        model_configs['device'] = args.device
        model_configs['ensemble_method'] = args.ensemble_method

    return model_configs