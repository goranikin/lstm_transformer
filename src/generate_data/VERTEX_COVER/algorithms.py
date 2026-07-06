from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.generate_data.common import ExternalSolverError
from src.generate_data.graph_utils import (
    adjacency_to_edges,
    is_vertex_cover,
    validate_adjacency,
)


class VertexCoverSolution(BaseModel):
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
) -> VertexCoverSolution:
    adjacency = validate_adjacency(adjacency)
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ExternalSolverError(
            "gurobipy is not installed. Install Gurobi's Python package before "
            "generating Vertex Cover labels."
        ) from exc

    num_nodes = adjacency.shape[0]
    model = gp.Model("minimum_vertex_cover")
    model.Params.OutputFlag = 0
    if seed is not None:
        model.Params.Seed = int(seed)

    x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(x[node] for node in range(num_nodes)), GRB.MINIMIZE)
    for u, v in adjacency_to_edges(adjacency):
        model.addConstr(x[u] + x[v] >= 1)

    model.optimize()
    if model.SolCount == 0:
        raise ExternalSolverError(
            f"Gurobi did not find a feasible vertex-cover solution: {model.Status}"
        )

    nodes = [node for node in range(num_nodes) if x[node].X > 0.5]
    if not is_vertex_cover(adjacency, nodes):
        raise ExternalSolverError("Gurobi produced an invalid vertex cover")

    is_exact = model.Status == GRB.OPTIMAL
    return VertexCoverSolution(
        algorithm="gurobi",
        nodes=nodes,
        size=len(nodes),
        is_exact=is_exact,
        metadata={
            "objective_bound": float(model.ObjBound),
            "status": int(model.Status),
        },
    )
