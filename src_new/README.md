# src_new

`src_new` is a clean experiment stack for modular neural combinatorial
optimization. It is separate from the old `src` package and is designed around
one interface shared by all encoders, decoders, and problem wrappers.

## Design

The key idea is that model components stay problem-agnostic:

- Encoders turn problem features into `EncoderOutput`.
- Decoders choose actions or scores through one `decode(...)` interface.
- Problem wrappers own masks, transitions, objectives, feasibility checks,
  supervised targets, and sigmoid repair/postprocessing.

This keeps TSP, CVRP, orienteering, knapsack, MIS, max clique, and vertex cover
behind the same training and inference path.

## Package Layout

```text
src_new/
  constants.py              Shared problem/model/mode names and defaults
  core.py                   Dataclasses for EncoderOutput, ProblemState, SolutionOutput
  data.py                   JSONL dataset adapter and batch collation
  model.py                  NCOModel wrapper combining one encoder, decoder, and problem
  utils.py                  Seed, device, timing, and tensor movement helpers

  problems/
    base.py                 Abstract Problem interface
    routing.py              TSPProblem and CVRPProblem
    subset.py               OrienteeringProblem and KnapsackProblem
    graph.py                MIS, max clique, and vertex cover wrappers
    registry.py             Problem factory

  models/
    encoders.py             AttentionEncoder and LSTMEncoder
    decoders.py             Attention, LSTM, GRU pointer decoders and sigmoid decoder

  training/
    trainer.py              Supervised and RL training loop
    baselines.py            Exponential and rollout reward baselines
    metrics.py              Objective, gap, feasibility, time, and seed variance helpers

  experiments/
    run.py                  Run one experiment
    matrix.py               Expand/run architecture matrix experiments
```

## Interfaces

Every encoder returns:

```python
EncoderOutput(
    node_embeddings=...,  # Tensor[B, N, d_model]
    graph_embedding=...,  # Tensor[B, d_model]
)
```

Every decoder returns:

```python
SolutionOutput(
    actions=...,
    log_probs=...,
    selected_mask=...,
    objective=...,
    feasible=...,
    reward=...,
)
```

Every problem wrapper implements:

```python
make_state()
get_mask()
step()
is_done()
compute_objective()
get_supervised_target()
repair_solution()
```

## Architecture Matrix

Encoders:

- `attention`
- `lstm`
- `graph_attention` currently aliases attention-style encoding

Decoders:

- `attention_pointer`
- `lstm_pointer`
- `gru_pointer`
- `sigmoid_subset`

The primary experiment matrix is:

```text
2 encoders x 4 decoders x 7 problems x 2 modes x 3 seeds = 336 runs
```

Use `src_new.experiments.matrix` to stage or dry-run those commands.

## Training Modes

Supervised learning uses solver-generated labels:

- Autoregressive decoders: negative log likelihood over target action sequence.
- Sigmoid decoder: `BCEWithLogitsLoss` against a binary target mask.

Reinforcement learning uses objective-based reward:

- Minimization: `reward = -objective`
- Maximization: `reward = objective`
- Baselines: `rollout` or `exponential`

## Metrics

Validation/test output tracks:

- Objective value
- Gap vs solver label when target objective exists
- Feasibility rate
- Inference time
- Training time
- Validation history curve

Seed variance is implemented as a helper in `src_new.training.metrics`; aggregate
multiple `result.json` files to compute final seed summaries.
