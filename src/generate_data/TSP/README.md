# TSP data generation

TSP instances are solved with [Concorde](https://www.math.uwaterloo.ca/tsp/concorde/), an
exact TSP solver. The `concorde` optional dependency installs
[PyConcorde](https://github.com/jvkersch/pyconcorde), which builds Concorde and
QSopt and exposes Concorde through Python.

## Quick check

```bash
uv sync --extra concorde
uv run python -c "from concorde.tsp import TSPSolver; print(TSPSolver)"
```

If the PyConcorde build fails on your system, install the standalone Concorde
binary (below) and point the generator at the executable using one of:

1. Put the binary on `PATH` as `concorde`.
2. Set `CONCORDE_EXECUTABLE=/path/to/concorde`.
3. Pass `--concorde-executable /path/to/concorde` to the generator.

Example:

```bash
export CONCORDE_EXECUTABLE=/opt/concorde/TSP/concorde

uv run python -m src.generate_data.TSP.generate \
  --num-instances 10 \
  --num-nodes 50 \
  --seed 1234 \
  --output-path data/tsp/tsp50_smoke.jsonl
```

## Install on Linux (recommended: prebuilt binary)

Academic use only. Download from the
[Concorde downloads page](https://www.math.uwaterloo.ca/tsp/concorde/downloads/downloads.htm).

```bash
mkdir -p ~/opt/concorde && cd ~/opt/concorde
curl -LO https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/concorde-linux.gz
gunzip concorde-linux.gz
chmod +x concorde-linux
mv concorde-linux concorde
export PATH="$HOME/opt/concorde:$PATH"
```

If the prebuilt binary does not run on your system (glibc mismatch), build from
source instead.

## Install on Linux (build from source)

You need Concorde source plus an LP solver (QSopt is the usual choice for
academic builds).

1. Download `Concorde-03.12.19` and QSopt from the
   [Concorde downloads page](https://www.math.uwaterloo.ca/tsp/concorde/downloads/downloads.htm).
2. Build QSopt, then configure and build Concorde with `--with-qsopt=...`.
3. The solver binary is `TSP/concorde` inside the build tree.

See the upstream
[README and installation guide](https://www.math.uwaterloo.ca/tsp/concorde/DOC/README.html)
for full details.

## Install on macOS

Build from source (same QSopt + Concorde steps as Linux), or use a Linux VM /
remote machine for dataset generation.

## Generator CLI

```bash
uv run python -m src.generate_data.TSP.generate --help
```

Batch commands for train/val/test splits are in
[COMMANDS.md](../COMMANDS.md).
