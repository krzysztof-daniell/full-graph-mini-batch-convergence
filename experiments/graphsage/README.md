# Best Results
## Mini-Batch
### Accuracy Convergence
```

```
### Speed Convergence
```
python mini_batch.py \
  --dataset ogbn-products
  --graph-self-loop \
  --lr 0.0005290823849783164 \
  --hidden-feats 384 \
  --num-layers 3 \
  --aggregator-type mean \
  --no-batch-norm \
  --input-dropout 0.10364511941698805 \
  --dropout 0.23607610901695883 \
  --activation leaky_relu \
  --batch-size 825 \
  --fanouts 4 3 3 \
  --test-validation \
  --seed 13
```
## Mini-Batch
### Accuracy Convergence
```

```
### Speed Convergence
```
python mini_batch.py \
  --dataset ogbn-products
  --graph-self-loop \
  --lr 0.011643816214941283 \
  --hidden-feats 384 \
  --num-layers 3 \
  --aggregator-type gcn \
  --batch-norm \
  --input-dropout 0.21698492695836674 \
  --dropout 0.33766791177553823 \
  --activation leaky_relu \
  --test-validation \
  --seed 13
```