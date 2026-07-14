# Experiment analysis

Run the analysis after exporting W&B histories:

```bash
uv run python -m src.analyze
```

The default input is `outputs/wandb_export/supervised_seed1234`. Results are
written to `outputs/analysis/supervised_seed1234`:

- `report.md`: concise architecture and cross-category interpretation.
- `architectures.csv`: exact dimensions and encoder/decoder parameter counts.
- `final_gaps.csv`: final label objectives, decoder objectives, normalized gaps,
  ranks, feasibility, and quality flags.
- `pairwise_decoder_comparisons.csv`: every decoder pair on every problem.
- `decoder_summary.csv`: cross-problem means, ranks, wins, and family profiles.
- `gap_trajectories.csv`: validation gap at each epoch.
- `analysis.json`: machine-readable consolidated results.

The primary metric is `aggregate_gap_pct`. It normalizes the difference between
the aggregate decoder objective and aggregate solver-label objective, respecting
whether a problem is minimized or maximized. The raw W&B percentage is retained
as `logged_gap_pct` because it answers a different question (mean per-instance
relative gap) and may be unstable when individual label objectives are zero.
