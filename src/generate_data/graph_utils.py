import numpy as np


def validate_adjacency(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=np.bool_)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if np.any(np.diag(adjacency)):
        raise ValueError("adjacency must not contain self-loops")
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("adjacency must be symmetric")
    return adjacency


def edges_to_adjacency(num_nodes: int, edges: list[tuple[int, int]]) -> np.ndarray:
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.bool_)
    for u, v in edges:
        if u == v:
            raise ValueError("graph cannot contain self-loops")
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError("edge endpoint out of range")
        adjacency[u, v] = True
        adjacency[v, u] = True
    return adjacency


def adjacency_to_edges(adjacency: np.ndarray) -> list[tuple[int, int]]:
    adjacency = validate_adjacency(adjacency)
    edges: list[tuple[int, int]] = []
    for u in range(adjacency.shape[0]):
        for v in range(u + 1, adjacency.shape[1]):
            if bool(adjacency[u, v]):
                edges.append((u, v))
    return edges


def generate_erdos_renyi_graph(
    num_nodes: int,
    edge_probability: float,
    seed: int,
) -> np.ndarray:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError("edge_probability must be in [0, 1]")
    rng = np.random.default_rng(seed)
    upper = rng.random((num_nodes, num_nodes)) < edge_probability
    upper = np.triu(upper, k=1)
    adjacency = upper | upper.T
    return adjacency.astype(np.bool_)


def is_clique(adjacency: np.ndarray, nodes: list[int]) -> bool:
    adjacency = validate_adjacency(adjacency)
    if len(nodes) != len(set(nodes)):
        return False
    for node in nodes:
        if not 0 <= node < adjacency.shape[0]:
            return False
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            if not bool(adjacency[u, v]):
                return False
    return True


def is_independent_set(adjacency: np.ndarray, nodes: list[int]) -> bool:
    adjacency = validate_adjacency(adjacency)
    if len(nodes) != len(set(nodes)):
        return False
    for node in nodes:
        if not 0 <= node < adjacency.shape[0]:
            return False
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            if bool(adjacency[u, v]):
                return False
    return True


def is_vertex_cover(adjacency: np.ndarray, nodes: list[int]) -> bool:
    adjacency = validate_adjacency(adjacency)
    cover = set(nodes)
    if len(cover) != len(nodes):
        return False
    if any(node < 0 or node >= adjacency.shape[0] for node in cover):
        return False
    for u, v in adjacency_to_edges(adjacency):
        if u not in cover and v not in cover:
            return False
    return True
