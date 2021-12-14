# Create an experiment
python create_experiment.py --sigopt-api-token <YOUR TOKEN> --model <GNN FLAVOR> --dataset <OGBN DATASET> --trainined-method <mini-batch or full-graph> --optimization-target <accuracy or speed>

# Launching Runs
python graphsage/mini_batch.py --experiment-id <PRINTED BY EXPERIMENT CREATION> --num-epochs <HOW MANY ITERATIONS> 
