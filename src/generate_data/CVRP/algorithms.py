from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.generate_data.common import ExternalSolverError


class CvrpSolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: str
    routes: list[list[int]]
    cost: float
    is_exact: bool
    metadata: dict[str, Any] | None = None

    def to_record(self) -> dict:
        record = {
            "algorithm": self.algorithm,
            "cost": self.cost,
            "is_exact": self.is_exact,
            "routes": self.routes,
        }
        if self.metadata:
            record["metadata"] = self.metadata
        return record


def validate_cvrp_inputs(
    depot: np.ndarray,
    coordinates: np.ndarray,
    demands: np.ndarray,
    vehicle_capacity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    depot = np.asarray(depot, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.int64)
    vehicle_capacity = int(vehicle_capacity)
    if depot.shape != (2,):
        raise ValueError("depot must have shape [2]")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [num_customers, 2]")
    if demands.ndim != 1 or len(demands) != len(coordinates):
        raise ValueError("demands must have shape [num_customers]")
    if len(coordinates) == 0:
        raise ValueError("at least one customer is required")
    if np.any(demands <= 0):
        raise ValueError("customer demands must be positive")
    if vehicle_capacity <= 0:
        raise ValueError("vehicle_capacity must be positive")
    if int(demands.max()) > vehicle_capacity:
        raise ValueError("vehicle_capacity must cover the largest demand")
    return depot, coordinates, demands, vehicle_capacity


def cvrp_route_cost(
    depot: np.ndarray,
    coordinates: np.ndarray,
    route: list[int],
) -> float:
    depot = np.asarray(depot, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if not route:
        return 0.0
    points = np.vstack([depot, coordinates[route], depot])
    edges = points[1:] - points[:-1]
    return float(np.linalg.norm(edges, axis=1).sum())


def cvrp_cost(
    depot: np.ndarray,
    coordinates: np.ndarray,
    routes: list[list[int]],
) -> float:
    return float(sum(cvrp_route_cost(depot, coordinates, route) for route in routes))


def validate_routes(
    routes: list[list[int]],
    demands: np.ndarray,
    vehicle_capacity: int,
) -> None:
    demands = np.asarray(demands, dtype=np.int64)
    seen: list[int] = []
    for route in routes:
        load = 0
        for customer in route:
            if customer < 0 or customer >= len(demands):
                raise RuntimeError("route contains out-of-range customer index")
            load += int(demands[customer])
            seen.append(customer)
        if load > vehicle_capacity:
            raise RuntimeError("route exceeds vehicle capacity")
    if sorted(seen) != list(range(len(demands))):
        raise RuntimeError("routes must visit every customer exactly once")


def solve_gurobi(
    depot: np.ndarray,
    coordinates: np.ndarray,
    demands: np.ndarray,
    vehicle_capacity: int,
    *,
    max_vehicles: int | None = None,
    seed: int | None = None,
    time_limit_sec: float | None = None,
) -> CvrpSolution:
    depot, coordinates, demands, vehicle_capacity = validate_cvrp_inputs(
        depot,
        coordinates,
        demands,
        vehicle_capacity,
    )
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ExternalSolverError(
            "gurobipy is not installed. Install Gurobi's Python package before "
            "generating CVRP labels."
        ) from exc

    num_customers = len(coordinates)
    nodes = range(num_customers + 1)
    customers = range(1, num_customers + 1)
    points = np.vstack([depot, coordinates])
    distances = {
        (i, j): float(np.linalg.norm(points[i] - points[j]))
        for i in nodes
        for j in nodes
        if i != j
    }

    model = gp.Model("capacitated_vehicle_routing")
    model.Params.OutputFlag = 0
    if seed is not None:
        model.Params.Seed = int(seed)
    if time_limit_sec is not None:
        model.Params.TimeLimit = float(time_limit_sec)

    x = model.addVars(distances.keys(), vtype=GRB.BINARY, name="x")
    load = model.addVars(
        customers,
        lb={customer: int(demands[customer - 1]) for customer in customers},
        ub=vehicle_capacity,
        vtype=GRB.CONTINUOUS,
        name="load",
    )

    model.setObjective(
        gp.quicksum(distances[i, j] * x[i, j] for i, j in distances),
        GRB.MINIMIZE,
    )
    for customer in customers:
        model.addConstr(
            gp.quicksum(x[i, customer] for i in nodes if i != customer) == 1
        )
        model.addConstr(
            gp.quicksum(x[customer, j] for j in nodes if j != customer) == 1
        )

    depot_out = gp.quicksum(x[0, j] for j in customers)
    depot_in = gp.quicksum(x[i, 0] for i in customers)
    model.addConstr(depot_out == depot_in)
    if max_vehicles is not None:
        model.addConstr(depot_out <= int(max_vehicles))

    for i in customers:
        for j in customers:
            if i == j:
                continue
            model.addConstr(
                load[j]
                >= load[i] + int(demands[j - 1]) - vehicle_capacity * (1 - x[i, j])
            )

    model.optimize()
    if model.SolCount == 0:
        raise ExternalSolverError(
            f"Gurobi did not find a feasible CVRP solution: {model.Status}"
        )

    routes = _extract_routes_from_arcs(
        {(i, j) for i, j in distances if x[i, j].X > 0.5},
        num_customers,
    )
    validate_routes(routes, demands, vehicle_capacity)
    is_exact = model.Status == GRB.OPTIMAL
    return CvrpSolution(
        algorithm="gurobi",
        routes=routes,
        cost=cvrp_cost(depot, coordinates, routes),
        is_exact=is_exact,
        metadata={
            "max_vehicles": max_vehicles,
            "objective_bound": float(model.ObjBound),
            "status": int(model.Status),
            "vehicle_capacity": vehicle_capacity,
        },
    )


def _extract_routes_from_arcs(
    arcs: set[tuple[int, int]],
    num_customers: int,
) -> list[list[int]]:
    successors: dict[int, list[int]] = {}
    for i, j in arcs:
        successors.setdefault(i, []).append(j)

    routes: list[list[int]] = []
    visited: set[int] = set()
    for start in sorted(successors.get(0, [])):
        route: list[int] = []
        current = start
        while current != 0:
            if current in visited:
                raise RuntimeError("Gurobi solution contains a repeated customer")
            visited.add(current)
            route.append(current - 1)
            next_nodes = successors.get(current, [])
            if len(next_nodes) != 1:
                raise RuntimeError("Gurobi solution has invalid route continuity")
            current = next_nodes[0]
        routes.append(route)

    if visited != set(range(1, num_customers + 1)):
        raise RuntimeError("Gurobi solution did not visit every customer")
    return routes
