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
  --stage routing \
  --seeds 1234 \
  --modes supervised \
  --epochs 1 \
  --steps-per-epoch 1 \
  --batch-size 8
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
  --format json \
  --output outputs/src/parameter_budget.json
```

Use an explicit target size:

```bash
uv run python -m src.experiments.parameter_comparison \
  --target-params 500000 \
  --min-d-model 32 \
  --max-d-model 512 \
  --d-model-step 8
```

Compare only one problem family:

```bash
uv run python -m src.experiments.parameter_comparison \
  --problems tsp,cvrp,orienteering
```

Include `graph_attention`, which currently shares the attention implementation:

```bash
uv run python -m src.experiments.parameter_comparison \
  --include-graph-attention
```

## Single Experiment

Run one supervised TSP experiment:

```bash
uv run python -m src.experiments.run \
  --problem tsp \
  --encoder attention \
  --decoder attention_pointer \
  --mode supervised \
  --train-path data/tsp/tsp50_train_64000_seed1234.jsonl \
  --val-path data/tsp/tsp50_val_10000_seed4321.jsonl \
  --test-path data/tsp/tsp50_test_10000_seed9999.jsonl \
  --target-algorithm concorde \
  --seed 1234 \
  --epochs 100 \
  --batch-size 512 \
  --output-dir outputs/src/single/tsp_attention_attention_pointer_sl
```

Run one RL knapsack experiment:

```bash
uv run python -m src.experiments.run \
  --problem knapsack \
  --encoder attention \
  --decoder sigmoid_subset \
  --mode rl \
  --train-path data/knapsack/knapsack100_train_64000_seed1234.jsonl \
  --val-path data/knapsack/knapsack100_val_10000_seed4321.jsonl \
  --test-path data/knapsack/knapsack100_test_10000_seed9999.jsonl \
  --target-algorithm dynamic_programming \
  --seed 1234 \
  --epochs 100 \
  --batch-size 512 \
  --baseline exponential \
  --output-dir outputs/src/single/knapsack_attention_sigmoid_rl
```

## Full Matrix

Dry-run the full staged matrix:

```bash
uv run python -m src.experiments.matrix --stage all
```

Execute the full matrix:

```bash
uv run python -m src.experiments.matrix \
  --stage all \
  --execute
```

This expands to:

```text
2 encoders x 4 decoders x 7 problems x 2 modes x 3 seeds = 336 runs
```

## Staged Execution

Routing problems only:

```bash
uv run python -m src.experiments.matrix \
  --stage routing \
  --execute
```

Subset problems only:

```bash
uv run python -m src.experiments.matrix \
  --stage subset \
  --execute
```

Orienteering only:

```bash
uv run python -m src.experiments.matrix \
  --stage hybrid \
  --execute
```

Skip weaker sigmoid routing baselines:

```bash
uv run python -m src.experiments.matrix \
  --stage all \
  --skip-sigmoid-routing \
  --execute
```

## Small Pilot

One seed, one mode, short training:

```bash
uv run python -m src.experiments.matrix \
  --stage all \
  --seeds 1234 \
  --modes supervised \
  --epochs 5 \
  --steps-per-epoch 100 \
  --batch-size 128 \
  --output-root outputs/src/pilot \
  --execute
```

Both modes for a subset of problems:

```bash
uv run python -m src.experiments.matrix \
  --problems tsp,knapsack,mis \
  --seeds 1234 \
  --modes supervised,rl \
  --epochs 10 \
  --steps-per-epoch 100 \
  --batch-size 128 \
  --output-root outputs/src/pilot_mixed \
  --execute
```

## Custom Architecture Slices

Attention encoder with all decoders:

```bash
uv run python -m src.experiments.matrix \
  --encoders attention \
  --decoders attention_pointer,lstm_pointer,gru_pointer,sigmoid_subset \
  --stage all
```

Pointer-Network-style recurrent encoder only:

```bash
uv run python -m src.experiments.matrix \
  --encoders lstm \
  --stage all
```

Sigmoid decoder only:

```bash
uv run python -m src.experiments.matrix \
  --decoders sigmoid_subset \
  --stage all
```

## Expected Data Paths

The matrix command expects this naming convention:

```text
data/
  tsp/
    tsp50_train_64000_seed1234.jsonl
    tsp50_val_10000_seed4321.jsonl
    tsp50_test_10000_seed9999.jsonl
  cvrp/
    cvrp50_train_64000_seed1234.jsonl
    cvrp50_val_10000_seed4321.jsonl
    cvrp50_test_10000_seed9999.jsonl
  orienteering/
    orienteering50_train_64000_seed1234.jsonl
    orienteering50_val_10000_seed4321.jsonl
    orienteering50_test_10000_seed9999.jsonl
  knapsack/
    knapsack100_train_64000_seed1234.jsonl
    knapsack100_val_10000_seed4321.jsonl
    knapsack100_test_10000_seed9999.jsonl
  mis/
    mis100_p015_train_64000_seed1234.jsonl
    mis100_p015_val_10000_seed4321.jsonl
    mis100_p015_test_10000_seed9999.jsonl
  max_clique/
    max_clique100_p050_train_64000_seed1234.jsonl
    max_clique100_p050_val_10000_seed4321.jsonl
    max_clique100_p050_test_10000_seed9999.jsonl
  vertex_cover/
    vertex_cover100_p015_train_64000_seed1234.jsonl
    vertex_cover100_p015_val_10000_seed4321.jsonl
    vertex_cover100_p015_test_10000_seed9999.jsonl
```

Override data root when needed:

```bash
uv run python -m src.experiments.matrix \
  --data-root /path/to/data \
  --output-root outputs/src/custom_data
```

## Outputs

Each run writes:

```text
result.json       Final arguments, history, training time, optional test metrics
history.json      Incremental validation curve and training history
last.pt           Last checkpoint, unless --no-checkpoints is set
best.pt           Best validation checkpoint, when validation is configured
```

For smoke/debug runs, disable checkpoints and progress bars:

```bash
uv run python -m src.experiments.run \
  --problem knapsack \
  --encoder lstm \
  --decoder sigmoid_subset \
  --mode supervised \
  --train-path data/knapsack/knapsack50_smoke_seed1234.jsonl \
  --target-algorithm dynamic_programming \
  --epochs 1 \
  --steps-per-epoch 1 \
  --batch-size 4 \
  --output-dir /tmp/src_knapsack_debug \
  --no-progress \
  --no-checkpoints
```
