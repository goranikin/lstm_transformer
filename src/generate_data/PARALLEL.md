# Parallel data generation (manual, 16-core server)

Run all commands from the repository root.

This guide chunks large splits so you can run all chunks for a split in **one
zellij tab**. Each chunk writes a `.partNN.jsonl` file. Append `&` to start
jobs in the background, then run `wait` to block until every chunk in that split
finishes. Merge the parts in order after each split completes.

## Setup

```bash
cd ~/projects/lstm_transformer
source .venv/bin/activate
uv sync --extra solvers

export GRB_THREADS=1
export GRB_LICENSE_FILE=$HOME/.gurobi/gurobi.lic
# export CONCORDE_EXECUTABLE=~/opt/concorde/concorde   # if needed
```

Recommended limits for a one-day run on 16 cores:

| Problem | `--solver-time-limit-sec` |
| ------- | ------------------------- |
| CVRP | `5` |
| Orienteering | `5` |
| Max Clique | `10` |
| MIS / Vertex Cover | omit (no limit) |

Train splits launch **8 parallel chunks**; val/test splits launch **2**. That
fits a 16-core machine. Detach from zellij with `Ctrl-g` then `d` while jobs
keep running. Check progress with `jobs -l` or `wc -l data/*/*.part*.jsonl`.

## Chunk layout

| Split | Total instances | Chunk size | Chunks | `--start-index` values |
| ----- | --------------: | ---------: | -----: | ---------------------- |
| Train | 64,000 | 8,000 | 8 | `0, 8000, ..., 56000` |
| Val / test | 10,000 | 5,000 | 2 | `0, 5000` |

Per-instance seeds stay `base_seed + index`, so chunks are reproducible and
disjoint.

## Merge commands

Run these only after **all** parts for a split have finished.

```bash
cat data/knapsack/knapsack100_seed1234.part{00..07}.jsonl > data/knapsack/knapsack100_seed1234.jsonl
cat data/knapsack/knapsack100_val_seed4321.part{00..01}.jsonl > data/knapsack/knapsack100_val_seed4321.jsonl
cat data/knapsack/knapsack100_test_seed9999.part{00..01}.jsonl > data/knapsack/knapsack100_test_seed9999.jsonl

cat data/tsp/tsp50_seed1234.part{00..07}.jsonl > data/tsp/tsp50_seed1234.jsonl
cat data/tsp/tsp50_val_seed4321.part{00..01}.jsonl > data/tsp/tsp50_val_seed4321.jsonl
cat data/tsp/tsp50_test_seed9999.part{00..01}.jsonl > data/tsp/tsp50_test_seed9999.jsonl

cat data/cvrp/cvrp50_seed1234.part{00..07}.jsonl > data/cvrp/cvrp50_seed1234.jsonl
cat data/cvrp/cvrp50_val_seed4321.part{00..01}.jsonl > data/cvrp/cvrp50_val_seed4321.jsonl
cat data/cvrp/cvrp50_test_seed9999.part{00..01}.jsonl > data/cvrp/cvrp50_test_seed9999.jsonl

cat data/mis/mis100_p015_seed1234.part{00..07}.jsonl > data/mis/mis100_p015_seed1234.jsonl
cat data/mis/mis100_p015_val_seed4321.part{00..01}.jsonl > data/mis/mis100_p015_val_seed4321.jsonl
cat data/mis/mis100_p015_test_seed9999.part{00..01}.jsonl > data/mis/mis100_p015_test_seed9999.jsonl

cat data/max_clique/max_clique100_p050_seed1234.part{00..07}.jsonl > data/max_clique/max_clique100_p050_seed1234.jsonl
cat data/max_clique/max_clique100_p050_val_seed4321.part{00..01}.jsonl > data/max_clique/max_clique100_p050_val_seed4321.jsonl
cat data/max_clique/max_clique100_p050_test_seed9999.part{00..01}.jsonl > data/max_clique/max_clique100_p050_test_seed9999.jsonl

cat data/vertex_cover/vertex_cover100_p015_seed1234.part{00..07}.jsonl > data/vertex_cover/vertex_cover100_p015_seed1234.jsonl
cat data/vertex_cover/vertex_cover100_p015_val_seed4321.part{00..01}.jsonl > data/vertex_cover/vertex_cover100_p015_val_seed4321.jsonl
cat data/vertex_cover/vertex_cover100_p015_test_seed9999.part{00..01}.jsonl > data/vertex_cover/vertex_cover100_p015_test_seed9999.jsonl

cat data/orienteering/orienteering50_seed1234.part{00..07}.jsonl > data/orienteering/orienteering50_seed1234.jsonl
cat data/orienteering/orienteering50_val_seed4321.part{00..01}.jsonl > data/orienteering/orienteering50_val_seed4321.jsonl
cat data/orienteering/orienteering50_test_seed9999.part{00..01}.jsonl > data/orienteering/orienteering50_test_seed9999.jsonl
```

Optional cleanup:

```bash
rm data/*/*.part*.jsonl
```

## Knapsack

### Train (`knapsack100_seed1234.jsonl`)

```bash
uv run python -m src.generate_data.KNAPSACK.generate --num-instances 8000 --start-index 0 --num-items 100 --seed 1234 --output-path data/knapsack/knapsack100_seed1234.part00.jsonl &
uv run python -m src.generate_data.KNAPSACK.generate --num-instances 8000 --start-index 8000 --num-items 100 --seed 1234 --output-path data/knapsack/knapsack100_seed1234.part01.jsonl &
uv run python -m src.generate_data.KNAPSACK.generate --num-instances 8000 --start-index 16000 --num-items 100 --seed 1234 --output-path data/knapsack/knapsack100_seed1234.part02.jsonl &
uv run python -m src.generate_data.KNAPSACK.generate --num-instances 8000 --start-index 24000 --num-items 100 --seed 1234 --output-path data/knapsack/knapsack100_seed1234.part03.jsonl &
uv run python -m src.generate_data.KNAPSACK.generate --num-instances 8000 --start-index 32000 --num-items 100 --seed 1234 --output-path data/knapsack/knapsack100_seed1234.part04.jsonl &
uv run python -m src.generate_data.KNAPSACK.generate --num-instances 8000 --start-index 40000 --num-items 100 --seed 1234 --output-path data/knapsack/knapsack100_seed1234.part05.jsonl &
uv run python -m src.generate_data.KNAPSACK.generate --num-instances 8000 --start-index 48000 --num-items 100 --seed 1234 --output-path data/knapsack/knapsack100_seed1234.part06.jsonl &
uv run python -m src.generate_data.KNAPSACK.generate --num-instances 8000 --start-index 56000 --num-items 100 --seed 1234 --output-path data/knapsack/knapsack100_seed1234.part07.jsonl &
wait
echo "Knapsack train chunks finished."
```

### Validation (`knapsack100_val_seed4321.jsonl`)

```bash
uv run python -m src.generate_data.KNAPSACK.generate --num-instances 5000 --start-index 0 --num-items 100 --seed 4321 --output-path data/knapsack/knapsack100_val_seed4321.part00.jsonl &
uv run python -m src.generate_data.KNAPSACK.generate --num-instances 5000 --start-index 5000 --num-items 100 --seed 4321 --output-path data/knapsack/knapsack100_val_seed4321.part01.jsonl &
wait
echo "Knapsack validation chunks finished."
```

### Test (`knapsack100_test_seed9999.jsonl`)

```bash
uv run python -m src.generate_data.KNAPSACK.generate --num-instances 5000 --start-index 0 --num-items 100 --seed 9999 --output-path data/knapsack/knapsack100_test_seed9999.part00.jsonl &
uv run python -m src.generate_data.KNAPSACK.generate --num-instances 5000 --start-index 5000 --num-items 100 --seed 9999 --output-path data/knapsack/knapsack100_test_seed9999.part01.jsonl &
wait
echo "Knapsack test chunks finished."
```

## TSP

### Train (`tsp50_seed1234.jsonl`)

```bash
uv run python -m src.generate_data.TSP.generate --num-instances 8000 --start-index 0 --num-nodes 50 --seed 1234 --output-path data/tsp/tsp50_seed1234.part00.jsonl &
uv run python -m src.generate_data.TSP.generate --num-instances 8000 --start-index 8000 --num-nodes 50 --seed 1234 --output-path data/tsp/tsp50_seed1234.part01.jsonl &
uv run python -m src.generate_data.TSP.generate --num-instances 8000 --start-index 16000 --num-nodes 50 --seed 1234 --output-path data/tsp/tsp50_seed1234.part02.jsonl &
uv run python -m src.generate_data.TSP.generate --num-instances 8000 --start-index 24000 --num-nodes 50 --seed 1234 --output-path data/tsp/tsp50_seed1234.part03.jsonl &
uv run python -m src.generate_data.TSP.generate --num-instances 8000 --start-index 32000 --num-nodes 50 --seed 1234 --output-path data/tsp/tsp50_seed1234.part04.jsonl &
uv run python -m src.generate_data.TSP.generate --num-instances 8000 --start-index 40000 --num-nodes 50 --seed 1234 --output-path data/tsp/tsp50_seed1234.part05.jsonl &
uv run python -m src.generate_data.TSP.generate --num-instances 8000 --start-index 48000 --num-nodes 50 --seed 1234 --output-path data/tsp/tsp50_seed1234.part06.jsonl &
uv run python -m src.generate_data.TSP.generate --num-instances 8000 --start-index 56000 --num-nodes 50 --seed 1234 --output-path data/tsp/tsp50_seed1234.part07.jsonl &
wait
echo "TSP train chunks finished."
```

### Validation (`tsp50_val_seed4321.jsonl`)

```bash
uv run python -m src.generate_data.TSP.generate --num-instances 5000 --start-index 0 --num-nodes 50 --seed 4321 --output-path data/tsp/tsp50_val_seed4321.part00.jsonl &
uv run python -m src.generate_data.TSP.generate --num-instances 5000 --start-index 5000 --num-nodes 50 --seed 4321 --output-path data/tsp/tsp50_val_seed4321.part01.jsonl &
wait
echo "TSP validation chunks finished."
```

### Test (`tsp50_test_seed9999.jsonl`)

```bash
uv run python -m src.generate_data.TSP.generate --num-instances 5000 --start-index 0 --num-nodes 50 --seed 9999 --output-path data/tsp/tsp50_test_seed9999.part00.jsonl &
uv run python -m src.generate_data.TSP.generate --num-instances 5000 --start-index 5000 --num-nodes 50 --seed 9999 --output-path data/tsp/tsp50_test_seed9999.part01.jsonl &
wait
echo "TSP test chunks finished."
```

## CVRP

Solver time limit: **5 seconds** per instance.

### Train (`cvrp50_seed1234.jsonl`)

```bash
uv run python -m src.generate_data.CVRP.generate --num-instances 8000 --start-index 0 --num-customers 50 --seed 1234 --solver-time-limit-sec 5 --output-path data/cvrp/cvrp50_seed1234.part00.jsonl &
uv run python -m src.generate_data.CVRP.generate --num-instances 8000 --start-index 8000 --num-customers 50 --seed 1234 --solver-time-limit-sec 5 --output-path data/cvrp/cvrp50_seed1234.part01.jsonl &
uv run python -m src.generate_data.CVRP.generate --num-instances 8000 --start-index 16000 --num-customers 50 --seed 1234 --solver-time-limit-sec 5 --output-path data/cvrp/cvrp50_seed1234.part02.jsonl &
uv run python -m src.generate_data.CVRP.generate --num-instances 8000 --start-index 24000 --num-customers 50 --seed 1234 --solver-time-limit-sec 5 --output-path data/cvrp/cvrp50_seed1234.part03.jsonl &
uv run python -m src.generate_data.CVRP.generate --num-instances 8000 --start-index 32000 --num-customers 50 --seed 1234 --solver-time-limit-sec 5 --output-path data/cvrp/cvrp50_seed1234.part04.jsonl &
uv run python -m src.generate_data.CVRP.generate --num-instances 8000 --start-index 40000 --num-customers 50 --seed 1234 --solver-time-limit-sec 5 --output-path data/cvrp/cvrp50_seed1234.part05.jsonl &
uv run python -m src.generate_data.CVRP.generate --num-instances 8000 --start-index 48000 --num-customers 50 --seed 1234 --solver-time-limit-sec 5 --output-path data/cvrp/cvrp50_seed1234.part06.jsonl &
uv run python -m src.generate_data.CVRP.generate --num-instances 8000 --start-index 56000 --num-customers 50 --seed 1234 --solver-time-limit-sec 5 --output-path data/cvrp/cvrp50_seed1234.part07.jsonl &
wait
echo "CVRP train chunks finished."
```

### Validation (`cvrp50_val_seed4321.jsonl`)

```bash
uv run python -m src.generate_data.CVRP.generate --num-instances 5000 --start-index 0 --num-customers 50 --seed 4321 --solver-time-limit-sec 5 --output-path data/cvrp/cvrp50_val_seed4321.part00.jsonl &
uv run python -m src.generate_data.CVRP.generate --num-instances 5000 --start-index 5000 --num-customers 50 --seed 4321 --solver-time-limit-sec 5 --output-path data/cvrp/cvrp50_val_seed4321.part01.jsonl &
wait
echo "CVRP validation chunks finished."
```

### Test (`cvrp50_test_seed9999.jsonl`)

```bash
uv run python -m src.generate_data.CVRP.generate --num-instances 5000 --start-index 0 --num-customers 50 --seed 9999 --solver-time-limit-sec 5 --output-path data/cvrp/cvrp50_test_seed9999.part00.jsonl &
uv run python -m src.generate_data.CVRP.generate --num-instances 5000 --start-index 5000 --num-customers 50 --seed 9999 --solver-time-limit-sec 5 --output-path data/cvrp/cvrp50_test_seed9999.part01.jsonl &
wait
echo "CVRP test chunks finished."
```

## MIS

### Train (`mis100_p015_seed1234.jsonl`)

```bash
uv run python -m src.generate_data.MIS.generate --num-instances 8000 --start-index 0 --num-nodes 100 --edge-probability 0.15 --seed 1234 --output-path data/mis/mis100_p015_seed1234.part00.jsonl &
uv run python -m src.generate_data.MIS.generate --num-instances 8000 --start-index 8000 --num-nodes 100 --edge-probability 0.15 --seed 1234 --output-path data/mis/mis100_p015_seed1234.part01.jsonl &
uv run python -m src.generate_data.MIS.generate --num-instances 8000 --start-index 16000 --num-nodes 100 --edge-probability 0.15 --seed 1234 --output-path data/mis/mis100_p015_seed1234.part02.jsonl &
uv run python -m src.generate_data.MIS.generate --num-instances 8000 --start-index 24000 --num-nodes 100 --edge-probability 0.15 --seed 1234 --output-path data/mis/mis100_p015_seed1234.part03.jsonl &
uv run python -m src.generate_data.MIS.generate --num-instances 8000 --start-index 32000 --num-nodes 100 --edge-probability 0.15 --seed 1234 --output-path data/mis/mis100_p015_seed1234.part04.jsonl &
uv run python -m src.generate_data.MIS.generate --num-instances 8000 --start-index 40000 --num-nodes 100 --edge-probability 0.15 --seed 1234 --output-path data/mis/mis100_p015_seed1234.part05.jsonl &
uv run python -m src.generate_data.MIS.generate --num-instances 8000 --start-index 48000 --num-nodes 100 --edge-probability 0.15 --seed 1234 --output-path data/mis/mis100_p015_seed1234.part06.jsonl &
uv run python -m src.generate_data.MIS.generate --num-instances 8000 --start-index 56000 --num-nodes 100 --edge-probability 0.15 --seed 1234 --output-path data/mis/mis100_p015_seed1234.part07.jsonl &
wait
echo "MIS train chunks finished."
```

### Validation (`mis100_p015_val_seed4321.jsonl`)

```bash
uv run python -m src.generate_data.MIS.generate --num-instances 5000 --start-index 0 --num-nodes 100 --edge-probability 0.15 --seed 4321 --output-path data/mis/mis100_p015_val_seed4321.part00.jsonl &
uv run python -m src.generate_data.MIS.generate --num-instances 5000 --start-index 5000 --num-nodes 100 --edge-probability 0.15 --seed 4321 --output-path data/mis/mis100_p015_val_seed4321.part01.jsonl &
wait
echo "MIS validation chunks finished."
```

### Test (`mis100_p015_test_seed9999.jsonl`)

```bash
uv run python -m src.generate_data.MIS.generate --num-instances 5000 --start-index 0 --num-nodes 100 --edge-probability 0.15 --seed 9999 --output-path data/mis/mis100_p015_test_seed9999.part00.jsonl &
uv run python -m src.generate_data.MIS.generate --num-instances 5000 --start-index 5000 --num-nodes 100 --edge-probability 0.15 --seed 9999 --output-path data/mis/mis100_p015_test_seed9999.part01.jsonl &
wait
echo "MIS test chunks finished."
```

## Maximum Clique

Solver time limit: **10 seconds** per instance.

### Train (`max_clique100_p050_seed1234.jsonl`)

```bash
uv run python -m src.generate_data.MAX_CLIQUE.generate --num-instances 8000 --start-index 0 --num-nodes 100 --edge-probability 0.5 --seed 1234 --solver-time-limit-sec 10 --output-path data/max_clique/max_clique100_p050_seed1234.part00.jsonl &
uv run python -m src.generate_data.MAX_CLIQUE.generate --num-instances 8000 --start-index 8000 --num-nodes 100 --edge-probability 0.5 --seed 1234 --solver-time-limit-sec 10 --output-path data/max_clique/max_clique100_p050_seed1234.part01.jsonl &
uv run python -m src.generate_data.MAX_CLIQUE.generate --num-instances 8000 --start-index 16000 --num-nodes 100 --edge-probability 0.5 --seed 1234 --solver-time-limit-sec 10 --output-path data/max_clique/max_clique100_p050_seed1234.part02.jsonl &
uv run python -m src.generate_data.MAX_CLIQUE.generate --num-instances 8000 --start-index 24000 --num-nodes 100 --edge-probability 0.5 --seed 1234 --solver-time-limit-sec 10 --output-path data/max_clique/max_clique100_p050_seed1234.part03.jsonl &
uv run python -m src.generate_data.MAX_CLIQUE.generate --num-instances 8000 --start-index 32000 --num-nodes 100 --edge-probability 0.5 --seed 1234 --solver-time-limit-sec 10 --output-path data/max_clique/max_clique100_p050_seed1234.part04.jsonl &
uv run python -m src.generate_data.MAX_CLIQUE.generate --num-instances 8000 --start-index 40000 --num-nodes 100 --edge-probability 0.5 --seed 1234 --solver-time-limit-sec 10 --output-path data/max_clique/max_clique100_p050_seed1234.part05.jsonl &
uv run python -m src.generate_data.MAX_CLIQUE.generate --num-instances 8000 --start-index 48000 --num-nodes 100 --edge-probability 0.5 --seed 1234 --solver-time-limit-sec 10 --output-path data/max_clique/max_clique100_p050_seed1234.part06.jsonl &
uv run python -m src.generate_data.MAX_CLIQUE.generate --num-instances 8000 --start-index 56000 --num-nodes 100 --edge-probability 0.5 --seed 1234 --solver-time-limit-sec 10 --output-path data/max_clique/max_clique100_p050_seed1234.part07.jsonl &
wait
echo "Max Clique train chunks finished."
```

### Validation (`max_clique100_p050_val_seed4321.jsonl`)

```bash
uv run python -m src.generate_data.MAX_CLIQUE.generate --num-instances 5000 --start-index 0 --num-nodes 100 --edge-probability 0.5 --seed 4321 --solver-time-limit-sec 10 --output-path data/max_clique/max_clique100_p050_val_seed4321.part00.jsonl &
uv run python -m src.generate_data.MAX_CLIQUE.generate --num-instances 5000 --start-index 5000 --num-nodes 100 --edge-probability 0.5 --seed 4321 --solver-time-limit-sec 10 --output-path data/max_clique/max_clique100_p050_val_seed4321.part01.jsonl &
wait
echo "Max Clique validation chunks finished."
```

### Test (`max_clique100_p050_test_seed9999.jsonl`)

```bash
uv run python -m src.generate_data.MAX_CLIQUE.generate --num-instances 5000 --start-index 0 --num-nodes 100 --edge-probability 0.5 --seed 9999 --solver-time-limit-sec 10 --output-path data/max_clique/max_clique100_p050_test_seed9999.part00.jsonl &
uv run python -m src.generate_data.MAX_CLIQUE.generate --num-instances 5000 --start-index 5000 --num-nodes 100 --edge-probability 0.5 --seed 9999 --solver-time-limit-sec 10 --output-path data/max_clique/max_clique100_p050_test_seed9999.part01.jsonl &
wait
echo "Max Clique test chunks finished."
```

## Minimum Vertex Cover

### Train (`vertex_cover100_p015_seed1234.jsonl`)

```bash
uv run python -m src.generate_data.VERTEX_COVER.generate --num-instances 8000 --start-index 0 --num-nodes 100 --edge-probability 0.15 --seed 1234 --output-path data/vertex_cover/vertex_cover100_p015_seed1234.part00.jsonl &
uv run python -m src.generate_data.VERTEX_COVER.generate --num-instances 8000 --start-index 8000 --num-nodes 100 --edge-probability 0.15 --seed 1234 --output-path data/vertex_cover/vertex_cover100_p015_seed1234.part01.jsonl &
uv run python -m src.generate_data.VERTEX_COVER.generate --num-instances 8000 --start-index 16000 --num-nodes 100 --edge-probability 0.15 --seed 1234 --output-path data/vertex_cover/vertex_cover100_p015_seed1234.part02.jsonl &
uv run python -m src.generate_data.VERTEX_COVER.generate --num-instances 8000 --start-index 24000 --num-nodes 100 --edge-probability 0.15 --seed 1234 --output-path data/vertex_cover/vertex_cover100_p015_seed1234.part03.jsonl &
uv run python -m src.generate_data.VERTEX_COVER.generate --num-instances 8000 --start-index 32000 --num-nodes 100 --edge-probability 0.15 --seed 1234 --output-path data/vertex_cover/vertex_cover100_p015_seed1234.part04.jsonl &
uv run python -m src.generate_data.VERTEX_COVER.generate --num-instances 8000 --start-index 40000 --num-nodes 100 --edge-probability 0.15 --seed 1234 --output-path data/vertex_cover/vertex_cover100_p015_seed1234.part05.jsonl &
uv run python -m src.generate_data.VERTEX_COVER.generate --num-instances 8000 --start-index 48000 --num-nodes 100 --edge-probability 0.15 --seed 1234 --output-path data/vertex_cover/vertex_cover100_p015_seed1234.part06.jsonl &
uv run python -m src.generate_data.VERTEX_COVER.generate --num-instances 8000 --start-index 56000 --num-nodes 100 --edge-probability 0.15 --seed 1234 --output-path data/vertex_cover/vertex_cover100_p015_seed1234.part07.jsonl &
wait
echo "Vertex Cover train chunks finished."
```

### Validation (`vertex_cover100_p015_val_seed4321.jsonl`)

```bash
uv run python -m src.generate_data.VERTEX_COVER.generate --num-instances 5000 --start-index 0 --num-nodes 100 --edge-probability 0.15 --seed 4321 --output-path data/vertex_cover/vertex_cover100_p015_val_seed4321.part00.jsonl &
uv run python -m src.generate_data.VERTEX_COVER.generate --num-instances 5000 --start-index 5000 --num-nodes 100 --edge-probability 0.15 --seed 4321 --output-path data/vertex_cover/vertex_cover100_p015_val_seed4321.part01.jsonl &
wait
echo "Vertex Cover validation chunks finished."
```

### Test (`vertex_cover100_p015_test_seed9999.jsonl`)

```bash
uv run python -m src.generate_data.VERTEX_COVER.generate --num-instances 5000 --start-index 0 --num-nodes 100 --edge-probability 0.15 --seed 9999 --output-path data/vertex_cover/vertex_cover100_p015_test_seed9999.part00.jsonl &
uv run python -m src.generate_data.VERTEX_COVER.generate --num-instances 5000 --start-index 5000 --num-nodes 100 --edge-probability 0.15 --seed 9999 --output-path data/vertex_cover/vertex_cover100_p015_test_seed9999.part01.jsonl &
wait
echo "Vertex Cover test chunks finished."
```

## Orienteering

Solver time limit: **5 seconds** per instance.

### Train (`orienteering50_seed1234.jsonl`)

```bash
uv run python -m src.generate_data.ORIENTEERING.generate --num-instances 8000 --start-index 0 --num-nodes 50 --seed 1234 --solver-time-limit-sec 5 --output-path data/orienteering/orienteering50_seed1234.part00.jsonl &
uv run python -m src.generate_data.ORIENTEERING.generate --num-instances 8000 --start-index 8000 --num-nodes 50 --seed 1234 --solver-time-limit-sec 5 --output-path data/orienteering/orienteering50_seed1234.part01.jsonl &
uv run python -m src.generate_data.ORIENTEERING.generate --num-instances 8000 --start-index 16000 --num-nodes 50 --seed 1234 --solver-time-limit-sec 5 --output-path data/orienteering/orienteering50_seed1234.part02.jsonl &
uv run python -m src.generate_data.ORIENTEERING.generate --num-instances 8000 --start-index 24000 --num-nodes 50 --seed 1234 --solver-time-limit-sec 5 --output-path data/orienteering/orienteering50_seed1234.part03.jsonl &
uv run python -m src.generate_data.ORIENTEERING.generate --num-instances 8000 --start-index 32000 --num-nodes 50 --seed 1234 --solver-time-limit-sec 5 --output-path data/orienteering/orienteering50_seed1234.part04.jsonl &
uv run python -m src.generate_data.ORIENTEERING.generate --num-instances 8000 --start-index 40000 --num-nodes 50 --seed 1234 --solver-time-limit-sec 5 --output-path data/orienteering/orienteering50_seed1234.part05.jsonl &
uv run python -m src.generate_data.ORIENTEERING.generate --num-instances 8000 --start-index 48000 --num-nodes 50 --seed 1234 --solver-time-limit-sec 5 --output-path data/orienteering/orienteering50_seed1234.part06.jsonl &
uv run python -m src.generate_data.ORIENTEERING.generate --num-instances 8000 --start-index 56000 --num-nodes 50 --seed 1234 --solver-time-limit-sec 5 --output-path data/orienteering/orienteering50_seed1234.part07.jsonl &
wait
echo "Orienteering train chunks finished."
```

### Validation (`orienteering50_val_seed4321.jsonl`)

```bash
uv run python -m src.generate_data.ORIENTEERING.generate --num-instances 5000 --start-index 0 --num-nodes 50 --seed 4321 --solver-time-limit-sec 5 --output-path data/orienteering/orienteering50_val_seed4321.part00.jsonl &
uv run python -m src.generate_data.ORIENTEERING.generate --num-instances 5000 --start-index 5000 --num-nodes 50 --seed 4321 --solver-time-limit-sec 5 --output-path data/orienteering/orienteering50_val_seed4321.part01.jsonl &
wait
echo "Orienteering validation chunks finished."
```

### Test (`orienteering50_test_seed9999.jsonl`)

```bash
uv run python -m src.generate_data.ORIENTEERING.generate --num-instances 5000 --start-index 0 --num-nodes 50 --seed 9999 --solver-time-limit-sec 5 --output-path data/orienteering/orienteering50_test_seed9999.part00.jsonl &
uv run python -m src.generate_data.ORIENTEERING.generate --num-instances 5000 --start-index 5000 --num-nodes 50 --seed 9999 --solver-time-limit-sec 5 --output-path data/orienteering/orienteering50_test_seed9999.part01.jsonl &
wait
echo "Orienteering test chunks finished."
```

## Suggested zellij workflow

1. Open `zellij --session datagen`.
2. Use **one tab per problem** (or per split for heavy problems).
3. Paste the full code block for that split into the tab.
4. After `wait` returns, run the matching `cat` merge command from above.
5. Keep a monitor tab running:

```bash
watch -n 2 'free -h; echo; ps -eo pid,comm,%mem,rss --sort=-rss | head -15'
```

6. Verify line counts at the end:

```bash
wc -l data/tsp/*.jsonl \
  data/cvrp/*.jsonl \
  data/mis/*.jsonl \
  data/knapsack/*.jsonl \
  data/max_clique/*.jsonl \
  data/vertex_cover/*.jsonl \
  data/orienteering/*.jsonl
```

Expected counts:

- `*_seed1234.jsonl` → `64000`
- `*_val_seed4321.jsonl` → `10000`
- `*_test_seed9999.jsonl` → `10000`
