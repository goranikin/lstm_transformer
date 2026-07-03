# MIS data generation

MIS instances are solved with [Gurobi](https://www.gurobi.com/) via `gurobipy`.
Install the Python package with:

```bash
uv sync --extra gurobi
```

You also need a valid Gurobi license on the machine running the generator.

## Quick check

```bash
uv run python -c "import gurobipy; print(gurobipy.gurobi.version())"
```

## Generator CLI

```bash
uv run python -m src.generate_data.MIS.generate --help
```

Batch commands for train/val/test splits are in
[COMMANDS.md](../COMMANDS.md).

Training configs should use `data.target_algorithm=gurobi` for MIS labels.
