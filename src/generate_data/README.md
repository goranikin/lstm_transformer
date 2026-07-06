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

Single-split commands are in [COMMANDS.md](COMMANDS.md). Chunked parallel runs
for a 16-core server are in [PARALLEL.md](PARALLEL.md).

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
