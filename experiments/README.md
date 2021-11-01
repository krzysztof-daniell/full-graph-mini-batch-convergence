# Installing the new SigOpt client
- pip install git+https://github.com/sigopt/sigopt-python@sj/main will provide the CLI and python client features
- log in to staging https://app.staging.sigopt.com/login, you will need to reset your password to get in
- get your API token from https://app.staging.sigopt.com/tokens/info, set it with export SIGOPT_API_TOKEN=x
- change your API url: export SIGOPT_API_URL=https://api-staging.sigopt.com
- change your APP url: export SIGOPT_APP_URL=https://app.staging.sigopt.com
- rough documentation index can be found here https://app.staging.sigopt.com/docs2/index, this would be a good place to start https://app.staging.sigopt.com/docs2/intro/overview
- the new UI can be found on here https://app.staging.sigopt.com
- Documentation still has some rough edges (the navigation on the left side of the page isn't ready yet) but you can use the rough index to navigate to any page https://app.staging.sigopt.com/docs2/index (edited)

# Create an experiment
python create_experiment.py --sigopt-api-token <YOUR TOKEN> --model <GNN FLAVOR> --dataset <OGBN DATASET> --trainined-method <mini-batch or full-graph> --optimization-target <accuracy or speed>

# Launching Runs
python graphsage/mini_batch.py --experiment-id <PRINTED BY EXPERIMENT CREATION> --num-epochs <HOW MANY ITERATIONS> 
