from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.generate_data.common import ExternalSolverError
from src.generate_data.graph_utils import is_clique, validate_adjacency


class MaxCliqueSolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: str
    nodes: list[int]
    size: int
    is_exact: bool
    metadata: dict[str, Any] | None = None

    def to_record(self) -> dict:
        record = {
            "algorithm": self.algorithm,
            "is_exact": self.is_exact,
            "nodes": self.nodes,
            "size": self.size,
        }
        if self.metadata:
            record["metadata"] = self.metadata
        return record


def solve_gurobi(
    adjacency: np.ndarray,
    *,
    seed: int | None = None,
) -> MaxCliqueSolution:
    adjacency = validate_adjacency(adjacency)
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ExternalSolverError(
            "gurobipy is not installed. Install Gurobi's Python package before "
            "generating Maximum Clique labels."
        ) from exc

    num_nodes = adjacency.shape[0]
    model = gp.Model("maximum_clique")
    model.Params.OutputFlag = 0
    if seed is not None:
        model.Params.Seed = int(seed)

    x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(x[node] for node in range(num_nodes)), GRB.MAXIMIZE)
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if not bool(adjacency[u, v]):
                model.addConstr(x[u] + x[v] <= 1)

    model.optimize()
    if model.SolCount == 0:
        raise ExternalSolverError(
            f"Gurobi did not find a feasible maximum-clique solution: {model.Status}"
        )

    nodes = [node for node in range(num_nodes) if x[node].X > 0.5]
    if not is_clique(adjacency, nodes):
        raise ExternalSolverError("Gurobi produced an invalid clique")

    is_exact = model.Status == GRB.OPTIMAL
    return MaxCliqueSolution(
        algorithm="gurobi",
        nodes=nodes,
        size=len(nodes),
        is_exact=is_exact,
        metadata={
            "objective_bound": float(model.ObjBound),
            "status": int(model.Status),
        },
    )
