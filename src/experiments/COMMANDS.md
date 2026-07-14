GPU 0 — routing + hybrid (90 runs)
tsp, cvrp, orienteering × 5 decoders × 2 modes × 3 seeds
```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m src.experiments.matrix \
  'problems=[tsp,cvrp,orienteering]' \
  execute=true
```
GPU 1 — subset (120 runs)
knapsack, mis, max_clique, vertex_cover × 5 decoders × 2 modes × 3 seeds
```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m src.experiments.matrix \
  stage=subset \
  data.graph_batch_size=2 \
  data.graph_eval_batch_size=2 \
  execute=true
```
