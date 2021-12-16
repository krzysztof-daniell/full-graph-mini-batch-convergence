# Best Results
## Mini-Batch
### Accuracy Convergence
```

```
### Speed Convergence
```
python mini_batch.py \
  --embedding-lr 0.035029889358481954 \
  --model-lr 0.0033638517089447143 \
  --hidden-feats 320 \
  --num-bases 2 \
  --num-layers 2 \
  --norm right \
  --layer-norm \
  --input-dropout 0.25 \
  --dropout 0.3486361608414741 \
  --activation relu \
  --self-loop \
  --batch-size 831 \
  --fanouts 11 6 \
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
  --embedding-lr 0.2019226890149828 \
  --model-lr 0.2019226890149828 \
  --hidden-feats 64 \
  --num-bases 4 \
  --num-layers 2 \
  --norm right \
  --no-layer-norm \
  --input-dropout 0.4075727675219478 \
  --dropout 0.2019226890149828 \
  --activation leaky_relu \
  --self-loop \
  --test-validation \
  --seed 13
```