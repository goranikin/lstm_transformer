# Data generation

Scripts and dataset loaders for problem instances written to JSONL files under
the repo-root `data/` directory.

| Problem | Module | Output docs |
|---------|--------|-------------|
| TSP | `src.generate_data.TSP` | [TSP/README.md](TSP/README.md) |
| MIS | `src.generate_data.MIS` | [MIS/README.md](MIS/README.md) |
| CVRP | `src.generate_data.CVRP` | routing total-set benchmark |
| Knapsack | `src.generate_data.KNAPSACK` | partial-subset benchmark |
| Maximum Clique | `src.generate_data.MAX_CLIQUE` | graph partial-subset benchmark |
| Minimum Vertex Cover | `src.generate_data.VERTEX_COVER` | graph partial-subset benchmark |
| Orienteering | `src.generate_data.ORIENTEERING` | hybrid subset-sequence benchmark |

Run generators as modules from the repo root, for example:

```bash
uv run python -m src.generate_data.TSP.generate --help
uv run python -m src.generate_data.MIS.generate --help
uv run python -m src.generate_data.CVRP.generate --help
uv run python -m src.generate_data.KNAPSACK.generate --help
uv run python -m src.generate_data.MAX_CLIQUE.generate --help
uv run python -m src.generate_data.VERTEX_COVER.generate --help
uv run python -m src.generate_data.ORIENTEERING.generate --help
```

Training code currently loads generated files via `TSPDataset` / `MISDataset` in
this package (see `src/training/utils.py`). The added benchmark generators also
include dataset loaders with the same file-backed JSONL pattern, but model and
trainer support for these new problem types should be wired through a problem
abstraction before using them in training.

Example generator commands:

```bash
uv run python -m src.generate_data.CVRP.generate \
  --num-instances 1000 \
  --num-customers 50 \
  --seed 1234 \
  --output-path data/cvrp/cvrp50_seed1234.jsonl

uv run python -m src.generate_data.KNAPSACK.generate \
  --num-instances 1000 \
  --num-items 100 \
  --seed 1234 \
  --output-path data/knapsack/knapsack100_seed1234.jsonl

uv run python -m src.generate_data.MAX_CLIQUE.generate \
  --num-instances 1000 \
  --num-nodes 100 \
  --edge-probability 0.5 \
  --seed 1234 \
  --output-path data/max_clique/max_clique100_p050_seed1234.jsonl

uv run python -m src.generate_data.VERTEX_COVER.generate \
  --num-instances 1000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 1234 \
  --output-path data/vertex_cover/vertex_cover100_p015_seed1234.jsonl

uv run python -m src.generate_data.ORIENTEERING.generate \
  --num-instances 1000 \
  --num-nodes 50 \
  --seed 1234 \
  --output-path data/orienteering/orienteering50_seed1234.jsonl
```

`CVRP` and `ORIENTEERING` use Gurobi with a default 30 second per-instance time
limit. `MAX_CLIQUE` and `VERTEX_COVER` also use Gurobi, without a default time
limit. Gurobi labels require `gurobipy` and a valid local license.
