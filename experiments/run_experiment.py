import argparse
import os


def run_experiment(args: argparse.ArgumentParser):
    path = f'{args.model}/{args.training_method.replace("-", "_")}.py'

    arguments = [f'--dataset {args.dataset}']

    if args.experiment_id is not None:
        arguments.append(f'--experiment-id {args.experiment_id}')

    if args.dataset == 'ogbn-arxiv':
        arguments.append('--graph-reverse-edges')

    if args.dataset in ['ogbn-products', 'ogbn-arxiv']:
        arguments.append('--graph-self-loop')

    if args.dataset_root is not None:
        arguments.append(f'--dataset-root {args.dataset_root}')

    if args.model != 'rgcn':
        arguments.append('--test-validation')
    else:
        arguments.append('--no-test-validation')

    if args.checkpoints_path is not None:
        arguments.append('--save-checkpoints-to-csv')
        arguments.append(f'--checkpoints-path {args.checkpoints_path}')

    arguments = ' '.join(arguments)

    env_vars = f'SIGOPT_API_TOKEN={args.sigopt_api_token}'

    if args.training_method == 'mini-batch':
        env_vars += ' OMP_NUM_THREADS=20'

    os.system(f'{env_vars} python {path} {arguments}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Create SigOpt experiment')

    argparser.add_argument('--sigopt-api-token', default=None, type=str)
    argparser.add_argument('--experiment-id', default=None, type=str)
    argparser.add_argument('--model', type=str,
                           choices=['gatv2', 'graphsage', 'rgcn'])
    argparser.add_argument('--dataset', type=str,
                           choices=['ogbn-arxiv', 'ogbn-mag', 'ogbn-products', 'ogbn-proteins'])
    argparser.add_argument('--dataset-root', default=None, type=str)
    argparser.add_argument('--training-method', type=str,
                           choices=['mini-batch', 'full-graph'])
    argparser.add_argument('--optimization-target', default=None, type=str,
                           choices=['accuracy', 'speed'])
    argparser.add_argument('--checkpoints-path', default=None, type=str)

    args = argparser.parse_args()

    run_experiment(args)
