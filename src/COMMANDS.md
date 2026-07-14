# src Experiment Commands

Run all commands from the repository root.

## Quick Checks

Compile the new package:

```bash
uv run python -m compileall -q src
```

Dry-run a small routing matrix:

```bash
uv run python -m src.experiments.matrix \
  stage=routing \
  'seeds=[1234]' \
  'modes=[supervised]' \
  trainer.epochs=1 \
  trainer.steps_per_epoch=1 \
  data.batch_size=8
```

## Parameter Matching

Compare trainable parameter counts and find per-architecture `d_model` / `d_ff`
settings that match a shared budget:

```bash
uv run python -m src.experiments.parameter_comparison
```

By default, the target is the largest parameter count among the base
`d_model=128` architectures. Write a JSON budget file:

```bash
uv run python -m src.experiments.parameter_comparison \
  format=json \
  output=outputs/src/parameter_budget.json
```

Use an explicit target size:

```bash
uv run python -m src.experiments.parameter_comparison \
  target_params=500000 \
  search.min_d_model=32 \
  search.max_d_model=512 \
  search.d_model_step=8
```

Compare only one problem family:

```bash
uv run python -m src.experiments.parameter_comparison \
  'problems=[tsp,cvrp,orienteering]'
```

Include `graph_attention`, which currently shares the attention implementation:

```bash
uv run python -m src.experiments.parameter_comparison \
  include_graph_attention=true
```

## Single Experiment

Run one supervised TSP experiment:

```bash
uv run python -m src.experiments.run \
  problem=tsp \
  encoder=attention \
  decoder=attention_pointer \
  mode=supervised \
  data.train_path=~/local_db/lstm_transformer/tsp/tsp50_train_64000.jsonl \
  data.val_path=~/local_db/lstm_transformer/tsp/tsp50_val_10000.jsonl \
  data.test_path=~/local_db/lstm_transformer/tsp/tsp50_test_10000.jsonl \
  data.target_algorithm=concorde \
  seed=1234 \
  trainer.epochs=100 \
  data.batch_size=512 \
  paths.output_dir=outputs/src/single/tsp_attention_attention_pointer_sl
```

Run one RL knapsack experiment:

```bash
uv run python -m src.experiments.run \
  problem=knapsack \
  encoder=attention \
  decoder=sigmoid_subset \
  mode=rl \
  data.train_path=~/local_db/lstm_transformer/knapsack/knapsack100_train_64000.jsonl \
  data.val_path=~/local_db/lstm_transformer/knapsack/knapsack100_val_10000.jsonl \
  data.test_path=~/local_db/lstm_transformer/knapsack/knapsack100_test_10000.jsonl \
  data.target_algorithm=dynamic_programming \
  seed=1234 \
  trainer.epochs=100 \
  data.batch_size=512 \
  trainer.baseline=exponential \
  paths.output_dir=outputs/src/single/knapsack_attention_sigmoid_rl
```

The run entry point uses Hydra with `configs/train.yaml`. Parameter matching is
enabled by default:

```bash
parameter_budget.enabled=true
parameter_budget.path=outputs/src/parameter_budget.json
```

Disable matching and provide explicit dimensions:

```bash
uv run python -m src.experiments.run \
  problem=tsp \
  encoder=attention \
  decoder=attention_pointer \
  mode=supervised \
  parameter_budget.enabled=false \
  model.d_model=128 \
  model.d_ff=512
```

## Full Matrix

Dry-run the full staged matrix:

```bash
uv run python -m src.experiments.matrix stage=all
```

Execute the full matrix:

```bash
uv run python -m src.experiments.matrix \
  stage=all \
  execute=true
```

This expands to:

```text
1 encoder x 5 decoders x 7 problems x 2 modes x 3 seeds = 210 runs
```

## Staged Execution

Routing problems only:

```bash
uv run python -m src.experiments.matrix \
  stage=routing \
  execute=true
```

Subset problems only:

```bash
uv run python -m src.experiments.matrix \
  stage=subset \
  execute=true
```

Orienteering only:

```bash
uv run python -m src.experiments.matrix \
  stage=hybrid \
  execute=true
```

Skip weaker sigmoid routing baselines:

```bash
uv run python -m src.experiments.matrix \
  stage=all \
  skip_sigmoid_routing=true \
  execute=true
```

## Small Pilot

One seed, one mode, short training:

```bash
uv run python -m src.experiments.matrix \
  stage=all \
  'seeds=[1234]' \
  'modes=[supervised]' \
  trainer.epochs=5 \
  trainer.steps_per_epoch=100 \
  data.batch_size=128 \
  paths.output_root=outputs/src/pilot \
  execute=true
```

Both modes for a subset of problems:

```bash
uv run python -m src.experiments.matrix \
  'problems=[tsp,knapsack,mis]' \
  'seeds=[1234]' \
  'modes=[supervised,rl]' \
  trainer.epochs=10 \
  trainer.steps_per_epoch=100 \
  data.batch_size=128 \
  paths.output_root=outputs/src/pilot_mixed \
  execute=true
```

## Custom Architecture Slices

Attention encoder with all decoders:

```bash
uv run python -m src.experiments.matrix \
  'encoders=[attention]' \
  'decoders=[attention_pointer,lstm_pointer,gru_pointer,transformer_pointer,sigmoid_subset]' \
  stage=all
```

Transformer pointer only, with a one-layer causal decoder cell:

```bash
uv run python -m src.experiments.matrix \
  'decoders=[transformer_pointer]' \
  model.transformer_decoder_layers=1 \
  stage=all
```

Sigmoid decoder only:

```bash
uv run python -m src.experiments.matrix \
  'decoders=[sigmoid_subset]' \
  stage=all
```

## Expected Data Paths

The matrix command expects this naming convention:

```text
data/
  tsp/
    tsp50_train_64000.jsonl
    tsp50_val_10000.jsonl
    tsp50_test_10000.jsonl
  cvrp/
    cvrp50_train_64000.jsonl
    cvrp50_val_10000.jsonl
    cvrp50_test_10000.jsonl
  orienteering/
    orienteering50_train_64000.jsonl
    orienteering50_val_10000.jsonl
    orienteering50_test_10000.jsonl
  knapsack/
    knapsack100_train_64000.jsonl
    knapsack100_val_10000.jsonl
    knapsack100_test_10000.jsonl
  mis/
    mis100_p015_train_64000.jsonl
    mis100_p015_val_10000.jsonl
    mis100_p015_test_10000.jsonl
  max_clique/
    max_clique100_p050_train_64000.jsonl
    max_clique100_p050_val_10000.jsonl
    max_clique100_p050_test_10000.jsonl
  vertex_cover/
    vertex_cover100_p015_train_64000.jsonl
    vertex_cover100_p015_val_10000.jsonl
    vertex_cover100_p015_test_10000.jsonl
```

Override data root when needed:

```bash
uv run python -m src.experiments.matrix \
  data.root=/path/to/data \
  paths.output_root=outputs/src/custom_data
```

## Outputs

Each run writes:

```text
result.json       Final config, history, training time, optional test metrics
history.json      Incremental validation curve and training history
last.pt           Last checkpoint, unless trainer.save_checkpoints=false
best.pt           Best validation checkpoint, when validation is configured
```

For smoke/debug runs, disable checkpoints and progress bars:

```bash
uv run python -m src.experiments.run \
  problem=knapsack \
  encoder=attention \
  decoder=sigmoid_subset \
  mode=supervised \
  data.use_default_paths=false \
  data.train_path=~/local_db/lstm_transformer/knapsack/knapsack50_smoke_seed1234.jsonl \
  data.target_algorithm=dynamic_programming \
  trainer.epochs=1 \
  trainer.steps_per_epoch=1 \
  data.batch_size=4 \
  paths.output_dir=/tmp/src_knapsack_debug \
  trainer.progress_bar=false \
  trainer.save_checkpoints=false
```
