from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.generate_data.common import ExternalSolverError


class OrienteeringSolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: str
    tour: list[int]
    value: int
    length: float
    is_exact: bool
    metadata: dict[str, Any] | None = None

    def to_record(self) -> dict:
        record = {
            "algorithm": self.algorithm,
            "is_exact": self.is_exact,
            "length": self.length,
            "tour": self.tour,
            "value": self.value,
        }
        if self.metadata:
            record["metadata"] = self.metadata
        return record


def validate_orienteering_inputs(
    depot: np.ndarray,
    coordinates: np.ndarray,
    prizes: np.ndarray,
    travel_budget: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    depot = np.asarray(depot, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    prizes = np.asarray(prizes, dtype=np.int64)
    travel_budget = float(travel_budget)
    if depot.shape != (2,):
        raise ValueError("depot must have shape [2]")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [num_nodes, 2]")
    if prizes.ndim != 1 or len(prizes) != len(coordinates):
        raise ValueError("prizes must have shape [num_nodes]")
    if len(coordinates) == 0:
        raise ValueError("at least one node is required")
    if np.any(prizes < 0):
        raise ValueError("prizes must be non-negative")
    if travel_budget <= 0.0:
        raise ValueError("travel_budget must be positive")
    return depot, coordinates, prizes, travel_budget


def tour_length(
    depot: np.ndarray,
    coordinates: np.ndarray,
    tour: list[int],
) -> float:
    depot = np.asarray(depot, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if not tour:
        return 0.0
    points = np.vstack([depot, coordinates[tour], depot])
    edges = points[1:] - points[:-1]
    return float(np.linalg.norm(edges, axis=1).sum())


def validate_tour(
    tour: list[int],
    coordinates: np.ndarray,
    travel_budget: float,
) -> None:
    if len(tour) != len(set(tour)):
        raise RuntimeError("tour contains duplicate nodes")
    if any(node < 0 or node >= len(coordinates) for node in tour):
        raise RuntimeError("tour contains out-of-range node index")


def solve_gurobi(
    depot: np.ndarray,
    coordinates: np.ndarray,
    prizes: np.ndarray,
    travel_budget: float,
    *,
    seed: int | None = None,
    time_limit_sec: float | None = None,
) -> OrienteeringSolution:
    depot, coordinates, prizes, travel_budget = validate_orienteering_inputs(
        depot,
        coordinates,
        prizes,
        travel_budget,
    )
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ExternalSolverError(
            "gurobipy is not installed. Install Gurobi's Python package before "
            "generating Orienteering labels."
        ) from exc

    num_nodes = len(coordinates)
    nodes = range(num_nodes + 1)
    customers = range(1, num_nodes + 1)
    points = np.vstack([depot, coordinates])
    distances = {
        (i, j): float(np.linalg.norm(points[i] - points[j]))
        for i in nodes
        for j in nodes
        if i != j
    }

    model = gp.Model("orienteering")
    model.Params.OutputFlag = 0
    if seed is not None:
        model.Params.Seed = int(seed)
    if time_limit_sec is not None:
        model.Params.TimeLimit = float(time_limit_sec)

    x = model.addVars(distances.keys(), vtype=GRB.BINARY, name="x")
    y = model.addVars(customers, vtype=GRB.BINARY, name="y")
    order = model.addVars(customers, lb=0.0, ub=num_nodes, name="order")

    model.setObjective(
        gp.quicksum(int(prizes[node - 1]) * y[node] for node in customers),
        GRB.MAXIMIZE,
    )
    model.addConstr(
        gp.quicksum(distances[i, j] * x[i, j] for i, j in distances) <= travel_budget
    )
    model.addConstr(gp.quicksum(x[0, j] for j in customers) <= 1)
    model.addConstr(
        gp.quicksum(x[0, j] for j in customers)
        == gp.quicksum(x[i, 0] for i in customers)
    )

    for customer in customers:
        model.addConstr(
            gp.quicksum(x[i, customer] for i in nodes if i != customer) == y[customer]
        )
        model.addConstr(
            gp.quicksum(x[customer, j] for j in nodes if j != customer) == y[customer]
        )
        model.addConstr(order[customer] >= y[customer])
        model.addConstr(order[customer] <= num_nodes * y[customer])

    for i in customers:
        for j in customers:
            if i == j:
                continue
            model.addConstr(order[i] - order[j] + num_nodes * x[i, j] <= num_nodes - 1)

    model.optimize()
    if model.SolCount == 0:
        raise ExternalSolverError(
            f"Gurobi did not find a feasible Orienteering solution: {model.Status}"
        )

    tour = _extract_tour_from_arcs(
        {(i, j) for i, j in distances if x[i, j].X > 0.5},
        num_nodes,
    )
    length = tour_length(depot, coordinates, tour)
    if length > travel_budget + 1e-9:
        raise RuntimeError("Gurobi tour exceeds travel budget")
    validate_tour(tour, coordinates, travel_budget)
    is_exact = model.Status == GRB.OPTIMAL
    return OrienteeringSolution(
        algorithm="gurobi",
        tour=tour,
        value=int(prizes[tour].sum()) if tour else 0,
        length=length,
        is_exact=is_exact,
        metadata={
            "objective_bound": float(model.ObjBound),
            "status": int(model.Status),
            "travel_budget": travel_budget,
        },
    )


def _extract_tour_from_arcs(
    arcs: set[tuple[int, int]],
    num_nodes: int,
) -> list[int]:
    successors: dict[int, list[int]] = {}
    for i, j in arcs:
        successors.setdefault(i, []).append(j)
    starts = successors.get(0, [])
    if not starts:
        return []
    if len(starts) != 1:
        raise RuntimeError("Gurobi solution has more than one tour start")

    tour: list[int] = []
    visited: set[int] = set()
    current = starts[0]
    while current != 0:
        if current in visited:
            raise RuntimeError("Gurobi solution contains a repeated node")
        visited.add(current)
        tour.append(current - 1)
        next_nodes = successors.get(current, [])
        if len(next_nodes) != 1:
            raise RuntimeError("Gurobi solution has invalid tour continuity")
        current = next_nodes[0]
    if any(node < 0 or node >= num_nodes for node in tour):
        raise RuntimeError("Gurobi solution contains out-of-range node")
    return tour
