import argparse
import os

import sigopt
import yaml


def create_experiment(args: argparse.ArgumentParser) -> str:
    if args.sigopt_api_token is not None:
        token = args.sigopt_api_token
    else:
        token = os.getenv('SIGOPT_API_TOKEN')

        if token is None:
            raise ValueError(
                'SigOpt API token is not provided. Please provide it by '
                '--sigopt-api-token argument or set '
                'SIGOPT_API_TOKEN environment variable.'
            )

    dataset = args.dataset.replace('-', '_')
    training_method = args.training_method.replace('-', '_')

    path = f'./{args.model}/configs/{dataset}_{training_method}_{args.optimization_target}.yml'

    with open(path) as f:
        experiment_metadata = yaml.load(f, Loader=yaml.FullLoader)

    conn = sigopt.Connection(token)
    experiment = conn.experiments().create(**experiment_metadata)

    return experiment.id


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Create SigOpt experiment')
    
    argparser.add_argument('--sigopt-api-token', default=None, type=str)
    argparser.add_argument('--model', type=str,
                           choices=['gat', 'graphsage', 'rgcn'])
    argparser.add_argument('--dataset', type=str,
                           choices=['ogbn-arxiv', 'ogbn-mag', 'ogbn-products', 'ogbn-proteins'])
    argparser.add_argument('--training-method', type=str,
                           choices=['mini-batch', 'full-graph'])
    argparser.add_argument('--optimization-target', type=str,
                           choices=['accuracy', 'speed'])

    args = argparser.parse_args()

    experiment_id = create_experiment(args)

    print(experiment_id)
