from __future__ import annotations

from typing import Any

import torch

from src_new.core import ProblemState, SupervisedTarget
from src_new.problems.base import Problem


class GraphSubsetProblem(Problem):
    name = "mis"
    objective_sense = "max"

    def build_features(self, batch: dict[str, Any]):
        adjacency = self.require_tensor(batch, "adjacency").bool()
        degree = adjacency.float().sum(dim=-1) / max(adjacency.size(1) - 1, 1)
        return degree.unsqueeze(-1), adjacency.float(), None

    def make_state(self, batch: dict[str, Any]) -> ProblemState:
        adjacency = self.require_tensor(batch, "adjacency")
        batch_size, node_count, _ = adjacency.shape
        device = adjacency.device
        return ProblemState(
            batch=batch,
            selected_mask=torch.zeros(batch_size, node_count, dtype=torch.bool, device=device),
            done=torch.zeros(batch_size, dtype=torch.bool, device=device),
            prev_action=torch.full((batch_size,), -1, dtype=torch.long, device=device),
            first_action=torch.full((batch_size,), -1, dtype=torch.long, device=device),
            aux={"available": torch.ones(batch_size, node_count, dtype=torch.bool, device=device)},
        )

    def get_mask(self, state: ProblemState) -> torch.Tensor:
        stop = state.selected_mask.size(1)
        node_mask = ~state.aux["available"]
        no_available = node_mask.all(dim=1)
        state.done = state.done | no_available
        mask = torch.cat([node_mask, torch.zeros(state.batch_size, 1, dtype=torch.bool, device=state.device)], dim=1)
        return self.apply_done_mask(mask, state.done, stop)

    def step(self, state: ProblemState, action: torch.Tensor) -> ProblemState:
        active = ~state.done
        node_count = state.selected_mask.size(1)
        stop = node_count
        action = action.clamp(min=0, max=stop).long()
        rows = torch.arange(state.batch_size, device=state.device)
        node_active = active & (action != stop)
        if node_active.any():
            state.selected_mask[rows[node_active], action[node_active]] = True
            self._update_available(state, node_active, action)
        self.append_action(state, action)
        state.done = state.done | (active & (action == stop))
        return state

    def _update_available(
        self,
        state: ProblemState,
        node_active: torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        adjacency = self.require_tensor(state.batch, "adjacency").bool()
        rows = torch.arange(state.batch_size, device=state.device)
        active_rows = rows[node_active]
        active_actions = action[node_active]
        state.aux["available"][active_rows] &= ~adjacency[active_rows, active_actions]
        state.aux["available"][active_rows, active_actions] = False

    def is_done(self, state: ProblemState) -> torch.Tensor:
        return state.done

    def context_features(self, state: ProblemState) -> torch.Tensor:
        selected_ratio = state.selected_mask.float().mean(dim=1)
        available_ratio = state.aux["available"].float().mean(dim=1)
        return torch.stack([selected_ratio, available_ratio, torch.zeros_like(selected_ratio), state.done.float()], dim=1)

    def to_solution(self, state: ProblemState) -> dict[str, torch.Tensor]:
        return {"actions": self.stack_actions(state), "selected_mask": state.selected_mask}

    def compute_objective(self, batch: dict[str, Any], solution: dict[str, torch.Tensor]):
        del batch
        return solution["selected_mask"].float().sum(dim=1)

    def check_feasible(self, batch: dict[str, Any], solution: dict[str, torch.Tensor]):
        adjacency = self.require_tensor(batch, "adjacency").bool()
        selected = solution["selected_mask"].bool()
        feasible = torch.ones(selected.size(0), dtype=torch.bool, device=selected.device)
        for row in range(selected.size(0)):
            nodes = torch.nonzero(selected[row], as_tuple=False).flatten()
            if nodes.numel() <= 1:
                continue
            subgraph = adjacency[row][nodes][:, nodes]
            feasible[row] = not bool(torch.triu(subgraph, diagonal=1).any())
        return feasible

    def get_supervised_target(self, batch: dict[str, Any]) -> SupervisedTarget:
        return SupervisedTarget(actions=self.target_actions(batch), selected_mask=self.target_mask(batch))

    def repair_solution(
        self,
        batch: dict[str, Any],
        scores: torch.Tensor,
        proposed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        adjacency = self.require_tensor(batch, "adjacency").bool()
        selected = torch.zeros_like(scores, dtype=torch.bool)
        actions = []
        for row in range(scores.size(0)):
            available = torch.ones(scores.size(1), dtype=torch.bool, device=scores.device)
            row_actions: list[int] = []
            for node in _candidate_order(scores[row], proposed_mask[row] if proposed_mask is not None else None):
                if bool(available[node]):
                    selected[row, node] = True
                    row_actions.append(node)
                    available &= ~adjacency[row, node]
                    available[node] = False
            row_actions.append(scores.size(1))
            actions.append(_pad_actions(row_actions, scores.size(1) + 1, scores.device))
        return {"actions": torch.stack(actions), "selected_mask": selected}


class MaxCliqueProblem(GraphSubsetProblem):
    name = "max_clique"
    objective_sense = "max"

    def _update_available(
        self,
        state: ProblemState,
        node_active: torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        adjacency = self.require_tensor(state.batch, "adjacency").bool()
        rows = torch.arange(state.batch_size, device=state.device)
        active_rows = rows[node_active]
        active_actions = action[node_active]
        state.aux["available"][active_rows] &= adjacency[active_rows, active_actions]
        state.aux["available"] &= ~state.selected_mask

    def check_feasible(self, batch: dict[str, Any], solution: dict[str, torch.Tensor]):
        adjacency = self.require_tensor(batch, "adjacency").bool()
        selected = solution["selected_mask"].bool()
        feasible = torch.ones(selected.size(0), dtype=torch.bool, device=selected.device)
        for row in range(selected.size(0)):
            nodes = torch.nonzero(selected[row], as_tuple=False).flatten()
            if nodes.numel() <= 1:
                continue
            subgraph = adjacency[row][nodes][:, nodes]
            expected_edges = nodes.numel() * (nodes.numel() - 1) // 2
            feasible[row] = int(torch.triu(subgraph, diagonal=1).sum().item()) == expected_edges
        return feasible

    def repair_solution(
        self,
        batch: dict[str, Any],
        scores: torch.Tensor,
        proposed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        adjacency = self.require_tensor(batch, "adjacency").bool()
        selected = torch.zeros_like(scores, dtype=torch.bool)
        actions = []
        for row in range(scores.size(0)):
            available = torch.ones(scores.size(1), dtype=torch.bool, device=scores.device)
            row_actions: list[int] = []
            for node in _candidate_order(scores[row], proposed_mask[row] if proposed_mask is not None else None):
                if bool(available[node]):
                    selected[row, node] = True
                    row_actions.append(node)
                    available &= adjacency[row, node]
                    available &= ~selected[row]
            row_actions.append(scores.size(1))
            actions.append(_pad_actions(row_actions, scores.size(1) + 1, scores.device))
        return {"actions": torch.stack(actions), "selected_mask": selected}


class VertexCoverProblem(GraphSubsetProblem):
    name = "vertex_cover"
    objective_sense = "min"

    def make_state(self, batch: dict[str, Any]) -> ProblemState:
        state = super().make_state(batch)
        adjacency = self.require_tensor(batch, "adjacency").bool()
        state.aux["covered_edges"] = torch.zeros_like(adjacency, dtype=torch.bool)
        return state

    def get_mask(self, state: ProblemState) -> torch.Tensor:
        node_count = state.selected_mask.size(1)
        adjacency = self.require_tensor(state.batch, "adjacency").bool()
        uncovered = adjacency & ~state.aux["covered_edges"]
        all_covered = ~uncovered.any(dim=(1, 2))
        state.done = state.done | all_covered
        node_mask = state.selected_mask.clone()
        stop_mask = (~all_covered).unsqueeze(1)
        mask = torch.cat([node_mask, stop_mask], dim=1)
        return self.apply_done_mask(mask, state.done, node_count)

    def _update_available(
        self,
        state: ProblemState,
        node_active: torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        rows = torch.arange(state.batch_size, device=state.device)
        active_rows = rows[node_active]
        active_actions = action[node_active]
        state.aux["covered_edges"][active_rows, active_actions, :] = True
        state.aux["covered_edges"][active_rows, :, active_actions] = True
        state.aux["available"] &= ~state.selected_mask

    def check_feasible(self, batch: dict[str, Any], solution: dict[str, torch.Tensor]):
        adjacency = self.require_tensor(batch, "adjacency").bool()
        selected = solution["selected_mask"].bool()
        feasible = torch.ones(selected.size(0), dtype=torch.bool, device=selected.device)
        for row in range(selected.size(0)):
            covered = selected[row].unsqueeze(0) | selected[row].unsqueeze(1)
            feasible[row] = not bool((adjacency[row] & ~covered).any())
        return feasible

    def repair_solution(
        self,
        batch: dict[str, Any],
        scores: torch.Tensor,
        proposed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        adjacency = self.require_tensor(batch, "adjacency").bool()
        selected = (
            proposed_mask.clone().bool()
            if proposed_mask is not None
            else scores > 0
        )
        for row in range(scores.size(0)):
            for u in range(scores.size(1)):
                for v in range(u + 1, scores.size(1)):
                    if bool(adjacency[row, u, v]) and not (bool(selected[row, u]) or bool(selected[row, v])):
                        pick = u if scores[row, u] >= scores[row, v] else v
                        selected[row, pick] = True
            for node in scores[row].argsort().tolist():
                if not bool(selected[row, node]):
                    continue
                candidate = selected[row].clone()
                candidate[node] = False
                covered = candidate.unsqueeze(0) | candidate.unsqueeze(1)
                if not bool((adjacency[row] & ~covered).any()):
                    selected[row, node] = False
        actions = []
        for row in range(scores.size(0)):
            row_actions = torch.nonzero(selected[row], as_tuple=False).flatten().tolist()
            row_actions.append(scores.size(1))
            actions.append(_pad_actions(row_actions, scores.size(1) + 1, scores.device))
        return {"actions": torch.stack(actions), "selected_mask": selected}


def _candidate_order(scores: torch.Tensor, proposed_mask: torch.Tensor | None) -> list[int]:
    if proposed_mask is not None and proposed_mask.any():
        nodes = torch.nonzero(proposed_mask, as_tuple=False).flatten()
        return nodes[scores[nodes].argsort(descending=True)].tolist()
    return [int(node) for node in scores.argsort(descending=True).tolist()]


def _pad_actions(actions: list[int], length: int, device: torch.device) -> torch.Tensor:
    padded = torch.full((length,), -1, dtype=torch.long, device=device)
    padded[: len(actions)] = torch.tensor(actions, dtype=torch.long, device=device)
    return padded
