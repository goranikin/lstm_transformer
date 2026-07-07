# Data generation commands

Run all commands from the repository root.

## Split settings

| Split | Instances | Seed |
|-------|-----------|------|
| Train | 64,000 | 1234 |
| Validation | 10,000 | 4321 |
| Test | 10,000 | 9999 |

Per-instance seeds are `base_seed + index`, so each split is reproducible and
disjoint.

Output files are written under `~/local_db/lstm_transformer/<problem>/`. Parent directories are
created automatically. Omit `--output-path` to use the default
`<prefix>_<split>_<num-instances>.jsonl` name (split is inferred from `--seed`).

## Solvers

| Problem | Solver used | Time limits |
| ------- | ----------- | ----------- |
| TSP | `concorde` | |
| MIS | `Gurobi` | |
| CVRP | `Gurobi` | 30s |
| Knapsack | `dynamic_programming` | |
| Max Clique | `Gurobi` | |
| Vertex Cover | `Gurobi` | |
| Orienteering | `Gurobi` | 30s |

TSP requires its external solver to be installed. See
[TSP/README.md](TSP/README.md) for Concorde setup details. Gurobi-backed
generators require `uv sync --extra gurobi` and a valid Gurobi license.

Install Python-packaged solvers with:

```bash
uv sync --extra solvers
```

For Concorde only, use `uv sync --extra concorde`; for Gurobi only, use
`uv sync --extra gurobi`.

## Output paths

| Problem | Train | Validation | Test |
|---------|-------|------------|------|
| TSP | `~/local_db/lstm_transformer/tsp/tsp50_train_64000.jsonl` | `~/local_db/lstm_transformer/tsp/tsp50_val_10000.jsonl` | `~/local_db/lstm_transformer/tsp/tsp50_test_10000.jsonl` |
| CVRP | `~/local_db/lstm_transformer/cvrp/cvrp50_train_64000.jsonl` | `~/local_db/lstm_transformer/cvrp/cvrp50_val_10000.jsonl` | `~/local_db/lstm_transformer/cvrp/cvrp50_test_10000.jsonl` |
| MIS | `~/local_db/lstm_transformer/mis/mis100_p015_train_64000.jsonl` | `~/local_db/lstm_transformer/mis/mis100_p015_val_10000.jsonl` | `~/local_db/lstm_transformer/mis/mis100_p015_test_10000.jsonl` |
| Knapsack | `~/local_db/lstm_transformer/knapsack/knapsack100_train_64000.jsonl` | `~/local_db/lstm_transformer/knapsack/knapsack100_val_10000.jsonl` | `~/local_db/lstm_transformer/knapsack/knapsack100_test_10000.jsonl` |
| Maximum Clique | `~/local_db/lstm_transformer/max_clique/max_clique100_p050_train_64000.jsonl` | `~/local_db/lstm_transformer/max_clique/max_clique100_p050_val_10000.jsonl` | `~/local_db/lstm_transformer/max_clique/max_clique100_p050_test_10000.jsonl` |
| Minimum Vertex Cover | `~/local_db/lstm_transformer/vertex_cover/vertex_cover100_p015_train_64000.jsonl` | `~/local_db/lstm_transformer/vertex_cover/vertex_cover100_p015_val_10000.jsonl` | `~/local_db/lstm_transformer/vertex_cover/vertex_cover100_p015_test_10000.jsonl` |
| Orienteering | `~/local_db/lstm_transformer/orienteering/orienteering50_train_64000.jsonl` | `~/local_db/lstm_transformer/orienteering/orienteering50_val_10000.jsonl` | `~/local_db/lstm_transformer/orienteering/orienteering50_test_10000.jsonl` |

## TSP

Solver: `concorde`.

```bash
uv run python -m src.generate_data.TSP.generate \
  --num-instances 64000 \
  --num-nodes 50 \
  --seed 1234 \
  --output-path ~/local_db/lstm_transformer/tsp/tsp50_train_64000.jsonl && \
uv run python -m src.generate_data.TSP.generate \
  --num-instances 10000 \
  --num-nodes 50 \
  --seed 4321 \
  --output-path ~/local_db/lstm_transformer/tsp/tsp50_val_10000.jsonl && \
uv run python -m src.generate_data.TSP.generate \
  --num-instances 10000 \
  --num-nodes 50 \
  --seed 9999 \
  --output-path ~/local_db/lstm_transformer/tsp/tsp50_test_10000.jsonl
```

## CVRP

Solver: `Gurobi` (30s per-instance time limit).

```bash
uv run python -m src.generate_data.CVRP.generate \
  --num-instances 64000 \
  --num-customers 50 \
  --seed 1234 \
  --output-path ~/local_db/lstm_transformer/cvrp/cvrp50_train_64000.jsonl && \
uv run python -m src.generate_data.CVRP.generate \
  --num-instances 10000 \
  --num-customers 50 \
  --seed 4321 \
  --output-path ~/local_db/lstm_transformer/cvrp/cvrp50_val_10000.jsonl && \
uv run python -m src.generate_data.CVRP.generate \
  --num-instances 10000 \
  --num-customers 50 \
  --seed 9999 \
  --output-path ~/local_db/lstm_transformer/cvrp/cvrp50_test_10000.jsonl
```

## MIS

Solver: `Gurobi`.

```bash
uv run python -m src.generate_data.MIS.generate \
  --num-instances 64000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 1234 \
  --output-path ~/local_db/lstm_transformer/mis/mis100_p015_train_64000.jsonl && \
uv run python -m src.generate_data.MIS.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 4321 \
  --output-path ~/local_db/lstm_transformer/mis/mis100_p015_val_10000.jsonl && \
uv run python -m src.generate_data.MIS.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 9999 \
  --output-path ~/local_db/lstm_transformer/mis/mis100_p015_test_10000.jsonl
```

## Knapsack

Solver: `dynamic_programming`.

```bash
uv run python -m src.generate_data.KNAPSACK.generate \
  --num-instances 64000 \
  --num-items 100 \
  --seed 1234 \
  --output-path ~/local_db/lstm_transformer/knapsack/knapsack100_train_64000.jsonl && \
uv run python -m src.generate_data.KNAPSACK.generate \
  --num-instances 10000 \
  --num-items 100 \
  --seed 4321 \
  --output-path ~/local_db/lstm_transformer/knapsack/knapsack100_val_10000.jsonl && \
uv run python -m src.generate_data.KNAPSACK.generate \
  --num-instances 10000 \
  --num-items 100 \
  --seed 9999 \
  --output-path ~/local_db/lstm_transformer/knapsack/knapsack100_test_10000.jsonl
```

## Maximum Clique

Solver: `Gurobi`.

```bash
uv run python -m src.generate_data.MAX_CLIQUE.generate \
  --num-instances 64000 \
  --num-nodes 100 \
  --edge-probability 0.5 \
  --seed 1234 \
  --output-path ~/local_db/lstm_transformer/max_clique/max_clique100_p050_train_64000.jsonl && \
uv run python -m src.generate_data.MAX_CLIQUE.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.5 \
  --seed 4321 \
  --output-path ~/local_db/lstm_transformer/max_clique/max_clique100_p050_val_10000.jsonl && \
uv run python -m src.generate_data.MAX_CLIQUE.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.5 \
  --seed 9999 \
  --output-path ~/local_db/lstm_transformer/max_clique/max_clique100_p050_test_10000.jsonl
```

## Minimum Vertex Cover

Solver: `Gurobi`.

```bash
uv run python -m src.generate_data.VERTEX_COVER.generate \
  --num-instances 64000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 1234 \
  --output-path ~/local_db/lstm_transformer/vertex_cover/vertex_cover100_p015_train_64000.jsonl && \
uv run python -m src.generate_data.VERTEX_COVER.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 4321 \
  --output-path ~/local_db/lstm_transformer/vertex_cover/vertex_cover100_p015_val_10000.jsonl && \
uv run python -m src.generate_data.VERTEX_COVER.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 9999 \
  --output-path ~/local_db/lstm_transformer/vertex_cover/vertex_cover100_p015_test_10000.jsonl
```

## Orienteering

Solver: `Gurobi` (30s per-instance time limit).

```bash
uv run python -m src.generate_data.ORIENTEERING.generate \
  --num-instances 64000 \
  --num-nodes 50 \
  --seed 1234 \
  --output-path ~/local_db/lstm_transformer/orienteering/orienteering50_train_64000.jsonl && \
uv run python -m src.generate_data.ORIENTEERING.generate \
  --num-instances 10000 \
  --num-nodes 50 \
  --seed 4321 \
  --output-path ~/local_db/lstm_transformer/orienteering/orienteering50_val_10000.jsonl && \
uv run python -m src.generate_data.ORIENTEERING.generate \
  --num-instances 10000 \
  --num-nodes 50 \
  --seed 9999 \
  --output-path ~/local_db/lstm_transformer/orienteering/orienteering50_test_10000.jsonl
```

## Notes

- See the [Solvers](#solvers) table for label algorithms and time limits.
- Knapsack `dynamic_programming` labels scale with item count and capacity.
- For chunked parallel runs on a 16-core server, see [PARALLEL.md](PARALLEL.md).
- Training configs in this repo expect `paths.train` and `paths.val` to point at
  the train and validation files above, for example:

```bash
uv run python -m src.main.train \
  problem=tsp \
  mode=supervised \
  paths.train=~/local_db/lstm_transformer/tsp/tsp50_train_64000.jsonl \
  paths.val=~/local_db/lstm_transformer/tsp/tsp50_val_10000.jsonl \
  data.target_algorithm=concorde
```
