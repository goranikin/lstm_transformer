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

Output files are written under `data/<problem>/`. Parent directories are
created automatically.

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
| TSP | `data/tsp/tsp50_seed1234.jsonl` | `data/tsp/tsp50_val_seed4321.jsonl` | `data/tsp/tsp50_test_seed9999.jsonl` |
| CVRP | `data/cvrp/cvrp50_seed1234.jsonl` | `data/cvrp/cvrp50_val_seed4321.jsonl` | `data/cvrp/cvrp50_test_seed9999.jsonl` |
| MIS | `data/mis/mis100_p015_seed1234.jsonl` | `data/mis/mis100_p015_val_seed4321.jsonl` | `data/mis/mis100_p015_test_seed9999.jsonl` |
| Knapsack | `data/knapsack/knapsack100_seed1234.jsonl` | `data/knapsack/knapsack100_val_seed4321.jsonl` | `data/knapsack/knapsack100_test_seed9999.jsonl` |
| Maximum Clique | `data/max_clique/max_clique100_p050_seed1234.jsonl` | `data/max_clique/max_clique100_p050_val_seed4321.jsonl` | `data/max_clique/max_clique100_p050_test_seed9999.jsonl` |
| Minimum Vertex Cover | `data/vertex_cover/vertex_cover100_p015_seed1234.jsonl` | `data/vertex_cover/vertex_cover100_p015_val_seed4321.jsonl` | `data/vertex_cover/vertex_cover100_p015_test_seed9999.jsonl` |
| Orienteering | `data/orienteering/orienteering50_seed1234.jsonl` | `data/orienteering/orienteering50_val_seed4321.jsonl` | `data/orienteering/orienteering50_test_seed9999.jsonl` |

## TSP

Solver: `concorde`.

```bash
uv run python -m src.generate_data.TSP.generate \
  --num-instances 64000 \
  --num-nodes 50 \
  --seed 1234 \
  --output-path data/tsp/tsp50_seed1234.jsonl && \
uv run python -m src.generate_data.TSP.generate \
  --num-instances 10000 \
  --num-nodes 50 \
  --seed 4321 \
  --output-path data/tsp/tsp50_val_seed4321.jsonl && \
uv run python -m src.generate_data.TSP.generate \
  --num-instances 10000 \
  --num-nodes 50 \
  --seed 9999 \
  --output-path data/tsp/tsp50_test_seed9999.jsonl
```

## CVRP

Solver: `Gurobi` (30s per-instance time limit).

```bash
uv run python -m src.generate_data.CVRP.generate \
  --num-instances 64000 \
  --num-customers 50 \
  --seed 1234 \
  --output-path data/cvrp/cvrp50_seed1234.jsonl && \
uv run python -m src.generate_data.CVRP.generate \
  --num-instances 10000 \
  --num-customers 50 \
  --seed 4321 \
  --output-path data/cvrp/cvrp50_val_seed4321.jsonl && \
uv run python -m src.generate_data.CVRP.generate \
  --num-instances 10000 \
  --num-customers 50 \
  --seed 9999 \
  --output-path data/cvrp/cvrp50_test_seed9999.jsonl
```

## MIS

Solver: `Gurobi`.

```bash
uv run python -m src.generate_data.MIS.generate \
  --num-instances 64000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 1234 \
  --output-path data/mis/mis100_p015_seed1234.jsonl && \
uv run python -m src.generate_data.MIS.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 4321 \
  --output-path data/mis/mis100_p015_val_seed4321.jsonl && \
uv run python -m src.generate_data.MIS.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 9999 \
  --output-path data/mis/mis100_p015_test_seed9999.jsonl
```

## Knapsack

Solver: `dynamic_programming`.

```bash
uv run python -m src.generate_data.KNAPSACK.generate \
  --num-instances 64000 \
  --num-items 100 \
  --seed 1234 \
  --output-path data/knapsack/knapsack100_seed1234.jsonl && \
uv run python -m src.generate_data.KNAPSACK.generate \
  --num-instances 10000 \
  --num-items 100 \
  --seed 4321 \
  --output-path data/knapsack/knapsack100_val_seed4321.jsonl && \
uv run python -m src.generate_data.KNAPSACK.generate \
  --num-instances 10000 \
  --num-items 100 \
  --seed 9999 \
  --output-path data/knapsack/knapsack100_test_seed9999.jsonl
```

## Maximum Clique

Solver: `Gurobi`.

```bash
uv run python -m src.generate_data.MAX_CLIQUE.generate \
  --num-instances 64000 \
  --num-nodes 100 \
  --edge-probability 0.5 \
  --seed 1234 \
  --output-path data/max_clique/max_clique100_p050_seed1234.jsonl && \
uv run python -m src.generate_data.MAX_CLIQUE.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.5 \
  --seed 4321 \
  --output-path data/max_clique/max_clique100_p050_val_seed4321.jsonl && \
uv run python -m src.generate_data.MAX_CLIQUE.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.5 \
  --seed 9999 \
  --output-path data/max_clique/max_clique100_p050_test_seed9999.jsonl
```

## Minimum Vertex Cover

Solver: `Gurobi`.

```bash
uv run python -m src.generate_data.VERTEX_COVER.generate \
  --num-instances 64000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 1234 \
  --output-path data/vertex_cover/vertex_cover100_p015_seed1234.jsonl && \
uv run python -m src.generate_data.VERTEX_COVER.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 4321 \
  --output-path data/vertex_cover/vertex_cover100_p015_val_seed4321.jsonl && \
uv run python -m src.generate_data.VERTEX_COVER.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 9999 \
  --output-path data/vertex_cover/vertex_cover100_p015_test_seed9999.jsonl
```

## Orienteering

Solver: `Gurobi` (30s per-instance time limit).

```bash
uv run python -m src.generate_data.ORIENTEERING.generate \
  --num-instances 64000 \
  --num-nodes 50 \
  --seed 1234 \
  --output-path data/orienteering/orienteering50_seed1234.jsonl && \
uv run python -m src.generate_data.ORIENTEERING.generate \
  --num-instances 10000 \
  --num-nodes 50 \
  --seed 4321 \
  --output-path data/orienteering/orienteering50_val_seed4321.jsonl && \
uv run python -m src.generate_data.ORIENTEERING.generate \
  --num-instances 10000 \
  --num-nodes 50 \
  --seed 9999 \
  --output-path data/orienteering/orienteering50_test_seed9999.jsonl
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
  paths.train=data/tsp/tsp50_seed1234.jsonl \
  paths.val=data/tsp/tsp50_val_seed4321.jsonl \
  data.target_algorithm=concorde
```
