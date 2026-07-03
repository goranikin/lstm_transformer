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

Requires Concorde for labels. See [TSP/README.md](TSP/README.md).

```bash
uv run python -m src.data_generating.TSP.generate \
  --num-instances 64000 \
  --num-nodes 50 \
  --seed 1234 \
  --output-path data/tsp/tsp50_seed1234.jsonl

uv run python -m src.data_generating.TSP.generate \
  --num-instances 10000 \
  --num-nodes 50 \
  --seed 4321 \
  --output-path data/tsp/tsp50_val_seed4321.jsonl

uv run python -m src.data_generating.TSP.generate \
  --num-instances 10000 \
  --num-nodes 50 \
  --seed 9999 \
  --output-path data/tsp/tsp50_test_seed9999.jsonl
```

## CVRP

Uses Gurobi with a default 30 second per-instance time limit.

```bash
uv run python -m src.data_generating.CVRP.generate \
  --num-instances 64000 \
  --num-customers 50 \
  --seed 1234 \
  --output-path data/cvrp/cvrp50_seed1234.jsonl

uv run python -m src.data_generating.CVRP.generate \
  --num-instances 10000 \
  --num-customers 50 \
  --seed 4321 \
  --output-path data/cvrp/cvrp50_val_seed4321.jsonl

uv run python -m src.data_generating.CVRP.generate \
  --num-instances 10000 \
  --num-customers 50 \
  --seed 9999 \
  --output-path data/cvrp/cvrp50_test_seed9999.jsonl
```

## MIS

Requires KaMIS for labels. See [MIS/README.md](MIS/README.md).

```bash
uv run python -m src.data_generating.MIS.generate \
  --num-instances 64000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 1234 \
  --output-path data/mis/mis100_p015_seed1234.jsonl

uv run python -m src.data_generating.MIS.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 4321 \
  --output-path data/mis/mis100_p015_val_seed4321.jsonl

uv run python -m src.data_generating.MIS.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 9999 \
  --output-path data/mis/mis100_p015_test_seed9999.jsonl
```

## Knapsack

Uses exact `dynamic_programming` labels.

```bash
uv run python -m src.data_generating.KNAPSACK.generate \
  --num-instances 64000 \
  --num-items 100 \
  --seed 1234 \
  --output-path data/knapsack/knapsack100_seed1234.jsonl

uv run python -m src.data_generating.KNAPSACK.generate \
  --num-instances 10000 \
  --num-items 100 \
  --seed 4321 \
  --output-path data/knapsack/knapsack100_val_seed4321.jsonl

uv run python -m src.data_generating.KNAPSACK.generate \
  --num-instances 10000 \
  --num-items 100 \
  --seed 9999 \
  --output-path data/knapsack/knapsack100_test_seed9999.jsonl
```

## Maximum Clique

Uses Gurobi labels.

```bash
uv run python -m src.data_generating.MAX_CLIQUE.generate \
  --num-instances 64000 \
  --num-nodes 100 \
  --edge-probability 0.5 \
  --seed 1234 \
  --output-path data/max_clique/max_clique100_p050_seed1234.jsonl

uv run python -m src.data_generating.MAX_CLIQUE.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.5 \
  --seed 4321 \
  --output-path data/max_clique/max_clique100_p050_val_seed4321.jsonl

uv run python -m src.data_generating.MAX_CLIQUE.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.5 \
  --seed 9999 \
  --output-path data/max_clique/max_clique100_p050_test_seed9999.jsonl
```

## Minimum Vertex Cover

```bash
uv run python -m src.data_generating.VERTEX_COVER.generate \
  --num-instances 64000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 1234 \
  --output-path data/vertex_cover/vertex_cover100_p015_seed1234.jsonl

uv run python -m src.data_generating.VERTEX_COVER.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 4321 \
  --output-path data/vertex_cover/vertex_cover100_p015_val_seed4321.jsonl

uv run python -m src.data_generating.VERTEX_COVER.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 9999 \
  --output-path data/vertex_cover/vertex_cover100_p015_test_seed9999.jsonl
```

## Orienteering

Uses Gurobi with a default 30 second per-instance time limit.

```bash
uv run python -m src.data_generating.ORIENTEERING.generate \
  --num-instances 64000 \
  --num-nodes 50 \
  --seed 1234 \
  --output-path data/orienteering/orienteering50_seed1234.jsonl

uv run python -m src.data_generating.ORIENTEERING.generate \
  --num-instances 10000 \
  --num-nodes 50 \
  --seed 4321 \
  --output-path data/orienteering/orienteering50_val_seed4321.jsonl

uv run python -m src.data_generating.ORIENTEERING.generate \
  --num-instances 10000 \
  --num-nodes 50 \
  --seed 9999 \
  --output-path data/orienteering/orienteering50_test_seed9999.jsonl
```

## Notes

- TSP, MIS, CVRP, Maximum Clique, Vertex Cover, and Orienteering generation
  require their configured external solvers to produce labels.
- Knapsack `dynamic_programming` labels scale with item count and capacity.
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
