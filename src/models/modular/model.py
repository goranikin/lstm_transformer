from typing import cast

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from configs.validation import default_am_config, validate_am_config
from src.constants import (
    MAXIMIZE_PROBLEMS,
    MINIMIZE_PROBLEMS,
    PROBLEM_FEATURE_DIMS,
    SUBSET_PROBLEMS,
    TOTAL_PROBLEMS,
    DecodeType,
    DecoderKind,
    EncoderKind,
    ObjectiveSense,
    ProblemType,
)
from src.models.modular.common import (
    closed_route_length,
    init_linear,
    log_prob,
    select,
)
from src.models.modular.decoder import (
    AttentionPointerDecoder,
    AttentionSubsetDecoder,
    RecurrentPointerDecoder,
    RecurrentSubsetDecoder,
    SigmoidSubsetHead,
)
from src.models.modular.encoders import AttentionEncoder, LSTMEncoder


class ModularNCOModel(nn.Module):
    """Pipeline that combines modular encoders and decoders for NCO tasks.

    `ModularNCOModel` owns the problem-specific glue around the reusable blocks:
    it builds node features, chooses an encoder, selects the requested decoder,
    applies feasibility masks for each problem, and computes comparable
    objective values. The public training code can therefore instantiate the
    requested architecture by setting only `encoder_kind` and `decoder_kind`,
    while the same forward and `supervised_loss` methods work across total-set,
    partial-set, and hybrid subset-sequence problems.
    """

    def __init__(
        self,
        *,
        config: DictConfig | None = None,
        encoder_kind: EncoderKind = "attention",
        decoder_kind: DecoderKind = "attention_pointer",
        default_problem: ProblemType | None = None,
    ) -> None:
        super().__init__()
        self.config = config or default_am_config()
        validate_am_config(self.config)
        self.encoder_kind = encoder_kind
        self.decoder_kind = decoder_kind
        self.default_problem = default_problem
        self.decode_type: DecodeType = "sampling"
        d_h = self.config.d_h

        self.input_embeds = nn.ModuleDict(
            {
                problem: nn.Linear(input_dim, d_h)
                for problem, input_dim in PROBLEM_FEATURE_DIMS.items()
            }
        )
        for embed in self.input_embeds.values():
            init_linear(cast(nn.Linear, embed))

        if encoder_kind == "lstm":
            self.encoder: nn.Module = LSTMEncoder(d_h=d_h)
        elif encoder_kind in ("attention", "graph_attention"):
            self.encoder = AttentionEncoder(
                n_layers=self.config.n_layers,
                n_heads=self.config.n_heads,
                d_h=d_h,
                d_ff=self.config.d_ff,
                normalization=self.config.normalization,
            )
        else:
            raise ValueError(f"Unsupported encoder_kind: {encoder_kind}")

        self.attention_pointer = AttentionPointerDecoder(
            d_h=d_h,
            n_heads=self.config.n_heads,
            tanh_clip=self.config.tanh_clip,
        )
        self.lstm_pointer = RecurrentPointerDecoder(
            d_h=d_h,
            tanh_clip=self.config.tanh_clip,
            cell_kind="lstm",
        )
        self.gru_pointer = RecurrentPointerDecoder(
            d_h=d_h,
            tanh_clip=self.config.tanh_clip,
            cell_kind="gru",
        )
        self.attention_subset = AttentionSubsetDecoder(
            d_h=d_h,
            n_heads=self.config.n_heads,
            tanh_clip=self.config.tanh_clip,
        )
        self.lstm_subset = RecurrentSubsetDecoder(
            d_h=d_h,
            tanh_clip=self.config.tanh_clip,
            cell_kind="lstm",
        )
        self.gru_subset = RecurrentSubsetDecoder(
            d_h=d_h,
            tanh_clip=self.config.tanh_clip,
            cell_kind="gru",
        )
        self.sigmoid_subset = SigmoidSubsetHead(d_h=d_h)

    def set_decode_type(self, decode_type: DecodeType) -> None:
        if decode_type not in ("greedy", "sampling"):
            raise ValueError("decode_type must be 'greedy' or 'sampling'")
        self.decode_type = decode_type

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        problem: ProblemType | None = None,
        return_pi: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        problem = self._resolve_problem(batch, problem)
        if problem in TOTAL_PROBLEMS:
            solution, log_likelihood = self._decode_total(batch, problem)
        elif problem in SUBSET_PROBLEMS:
            solution, log_likelihood = self._decode_subset(batch, problem)
        else:
            raise ValueError(f"Unsupported problem: {problem}")
        value = self.solution_value(batch, solution, problem=problem)
        cost = value if self.objective_sense(problem) == "min" else -value
        if return_pi:
            return cost, log_likelihood, solution
        return cost, log_likelihood

    def supervised_loss(
        self,
        batch: dict[str, torch.Tensor],
        problem: ProblemType | None = None,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        problem = self._resolve_problem(batch, problem)
        h, h_bar = self._encode(batch, problem)
        if problem in TOTAL_PROBLEMS:
            target = self._require_tensor(batch, "target_tour").long()
            if self.decoder_kind == "sigmoid_subset":
                logits = self.sigmoid_subset(h)
                return self._static_order_loss(
                    logits,
                    target,
                    label_smoothing=label_smoothing,
                )
            if label_smoothing:
                raise ValueError(
                    "label_smoothing is only supported for modular sigmoid_subset "
                    "decoders"
                )
            _, log_likelihood = self._pointer_decoder()(
                h,
                h_bar,
                decode_type="greedy",
                target=target,
            )
            return -log_likelihood.mean() / max(target.size(1), 1)

        target_set = self._require_tensor(batch, "target_set").float()
        if self.decoder_kind == "sigmoid_subset":
            logits = self.sigmoid_subset(h)
            if label_smoothing:
                target_set = (
                    target_set * (1.0 - label_smoothing) + 0.5 * label_smoothing
                )
            return F.binary_cross_entropy_with_logits(logits, target_set)
        if label_smoothing:
            raise ValueError(
                "label_smoothing is only supported for modular sigmoid_subset decoders"
            )

        target_sequence = self._target_sequence(batch, target_set)
        _, log_likelihood = self._decode_subset_from_embeddings(
            h,
            h_bar,
            batch,
            problem,
            target_sequence=target_sequence,
        )
        return -log_likelihood.mean() / max(target_sequence.size(1), 1)

    @torch.no_grad()
    def solution_value(
        self,
        batch: dict[str, torch.Tensor],
        solution: torch.Tensor,
        problem: ProblemType | None = None,
    ) -> torch.Tensor:
        problem = self._resolve_problem(batch, problem)
        if problem == "tsp":
            loc = self._require_tensor(batch, "loc")
            return closed_route_length(loc, solution.long())
        if problem == "cvrp":
            return self._cvrp_cost(batch, solution.long())
        if problem in ("mis", "maximum_clique", "minimum_vertex_cover"):
            return solution.float().sum(dim=1)
        if problem == "knapsack":
            values = self._require_tensor(batch, "values").float()
            return (solution.float() * values).sum(dim=1)
        if problem == "orienteering":
            prizes = self._require_tensor(batch, "prizes").float()
            return (solution.float() * prizes).sum(dim=1)
        raise ValueError(f"Unsupported problem: {problem}")

    def target_value(
        self,
        batch: dict[str, torch.Tensor],
        problem: ProblemType | None = None,
    ) -> torch.Tensor:
        problem = self._resolve_problem(batch, problem)
        if problem in ("tsp", "cvrp"):
            target = self._require_tensor(batch, "target_cost")
            return target.float()
        if problem in ("mis", "maximum_clique", "minimum_vertex_cover"):
            target = self._require_tensor(batch, "target_size")
            return target.float()
        if problem in ("knapsack", "orienteering"):
            target = self._require_tensor(batch, "target_value")
            return target.float()
        raise ValueError(f"Unsupported problem: {problem}")

    @staticmethod
    def objective_sense(problem: ProblemType) -> ObjectiveSense:
        if problem in MINIMIZE_PROBLEMS:
            return "min"
        if problem in MAXIMIZE_PROBLEMS:
            return "max"
        raise ValueError(f"Unsupported problem: {problem}")

    def _decode_total(
        self,
        batch: dict[str, torch.Tensor],
        problem: ProblemType,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, h_bar = self._encode(batch, problem)
        if self.decoder_kind == "sigmoid_subset":
            logits = self.sigmoid_subset(h)
            order = logits.argsort(dim=1, descending=True)
            log_likelihood = torch.zeros(h.size(0), dtype=h.dtype, device=h.device)
            return order, log_likelihood
        return self._pointer_decoder()(h, h_bar, decode_type=self.decode_type)

    def _decode_subset(
        self,
        batch: dict[str, torch.Tensor],
        problem: ProblemType,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, h_bar = self._encode(batch, problem)
        if self.decoder_kind == "sigmoid_subset":
            logits = self.sigmoid_subset(h)
            solution = self._decode_sigmoid_solution(batch, logits, problem)
            log_likelihood = torch.zeros(h.size(0), dtype=h.dtype, device=h.device)
            return solution, log_likelihood
        return self._decode_subset_from_embeddings(h, h_bar, batch, problem)

    def _decode_subset_from_embeddings(
        self,
        h: torch.Tensor,
        h_bar: torch.Tensor,
        batch: dict[str, torch.Tensor],
        problem: ProblemType,
        *,
        target_sequence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = h.shape
        device = h.device
        stop_index = num_nodes
        selected_mask = torch.zeros(
            batch_size,
            num_nodes,
            dtype=torch.bool,
            device=device,
        )
        available = torch.ones_like(selected_mask)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        prev_emb = self._subset_prev_placeholder().expand(batch_size, -1)
        batch_idx = torch.arange(batch_size, device=device)
        steps = num_nodes + 1

        remaining_capacity = (
            self._require_tensor(batch, "capacity").float().clone()
            if problem == "knapsack"
            else None
        )
        adjacency = (
            self._require_tensor(batch, "adjacency").bool()
            if problem in ("mis", "maximum_clique", "minimum_vertex_cover")
            else None
        )
        covered_edges = (
            torch.zeros_like(adjacency, dtype=torch.bool)
            if problem == "minimum_vertex_cover" and adjacency is not None
            else None
        )
        current_loc = (
            self._require_tensor(batch, "depot").float().clone()
            if problem == "orienteering"
            else None
        )
        path_length = (
            torch.zeros(batch_size, dtype=h.dtype, device=device)
            if problem == "orienteering"
            else None
        )

        recurrent = self._recurrent_subset_decoder()
        hidden: torch.Tensor | None = None
        cell: torch.Tensor | None = None
        if recurrent is not None:
            hidden, cell = recurrent.initial_state(h_bar)

        log_p_steps: list[torch.Tensor] = []
        for step in range(steps):
            context = self._subset_context(selected_mask, available)
            action_mask = self._subset_action_mask(
                batch=batch,
                problem=problem,
                available=available,
                finished=finished,
                selected_mask=selected_mask,
                remaining_capacity=remaining_capacity,
                covered_edges=covered_edges,
                current_loc=current_loc,
                path_length=path_length,
            )
            if recurrent is None:
                logits = self.attention_subset.step(
                    h,
                    h_bar,
                    prev_emb,
                    context,
                    action_mask,
                )
            else:
                decoder_input = torch.cat([prev_emb, context], dim=-1)
                if hidden is None:
                    raise RuntimeError("subset recurrent state is not initialized")
                hidden, cell, logits = recurrent.step(
                    h,
                    hidden,
                    cell,
                    decoder_input,
                    action_mask,
                )

            if target_sequence is None:
                selected, log_p = select(logits, self.decode_type)
            else:
                selected = target_sequence[:, step]
                log_p = log_prob(logits, selected)
            selected = torch.where(
                finished,
                torch.full_like(selected, stop_index),
                selected,
            )
            log_p = torch.where(finished, torch.zeros_like(log_p), log_p)
            active = (~finished) & (selected != stop_index)
            node_selected = selected.clamp(max=num_nodes - 1)

            if active.any():
                selected_mask[active, node_selected[active]] = True
                prev_emb = torch.where(
                    active.unsqueeze(1),
                    h[batch_idx, node_selected],
                    prev_emb,
                )
                available = self._update_available(
                    batch=batch,
                    problem=problem,
                    available=available,
                    active=active,
                    node_selected=node_selected,
                    selected_mask=selected_mask,
                    remaining_capacity=remaining_capacity,
                    adjacency=adjacency,
                    covered_edges=covered_edges,
                    current_loc=current_loc,
                    path_length=path_length,
                )

            finished = finished | (selected == stop_index)
            log_p_steps.append(log_p)
            if target_sequence is None and finished.all():
                break

        return selected_mask, torch.stack(log_p_steps, dim=1).sum(dim=1)

    def _encode(
        self,
        batch: dict[str, torch.Tensor],
        problem: ProblemType,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._node_features(batch, problem)
        embed = cast(nn.Linear, self.input_embeds[problem])
        h0 = embed(features)
        mask = self._encoder_mask(batch, problem)
        h, h_bar = self.encoder(h0, mask=mask)
        return h, h_bar

    def _node_features(
        self,
        batch: dict[str, torch.Tensor],
        problem: ProblemType,
    ) -> torch.Tensor:
        if problem == "tsp":
            return self._require_tensor(batch, "loc").float()
        if problem == "cvrp":
            coordinates = self._require_tensor(batch, "coordinates").float()
            demands = self._require_tensor(batch, "demands").float()
            capacity = self._require_tensor(batch, "capacity").float().view(-1, 1, 1)
            return torch.cat([coordinates, demands.unsqueeze(-1) / capacity], dim=-1)
        if problem in ("mis", "maximum_clique", "minimum_vertex_cover"):
            adjacency = self._require_tensor(batch, "adjacency").bool()
            degree = adjacency.float().sum(dim=-1) / max(adjacency.size(1) - 1, 1)
            return degree.unsqueeze(-1)
        if problem == "knapsack":
            return self._require_tensor(batch, "item_features").float()
        if problem == "orienteering":
            coordinates = self._require_tensor(batch, "coordinates").float()
            prizes = self._require_tensor(batch, "prizes").float()
            scaled_prizes = prizes / prizes.max(dim=1, keepdim=True).values.clamp_min(
                1.0
            )
            return torch.cat([coordinates, scaled_prizes.unsqueeze(-1)], dim=-1)
        raise ValueError(f"Unsupported problem: {problem}")

    def _encoder_mask(
        self,
        batch: dict[str, torch.Tensor],
        problem: ProblemType,
    ) -> torch.Tensor | None:
        if self.encoder_kind != "graph_attention":
            return None
        if problem not in ("mis", "maximum_clique", "minimum_vertex_cover"):
            return None
        adjacency = self._require_tensor(batch, "adjacency").bool()
        mask = ~adjacency
        diag = torch.eye(adjacency.size(1), dtype=torch.bool, device=adjacency.device)
        mask[:, diag] = False
        return mask

    def _pointer_decoder(self) -> nn.Module:
        if self.decoder_kind == "attention_pointer":
            return self.attention_pointer
        if self.decoder_kind in ("lstm_pointer", "lstm_subset"):
            return self.lstm_pointer
        if self.decoder_kind in ("gru_pointer", "gru_subset"):
            return self.gru_pointer
        if self.decoder_kind == "attention_subset":
            return self.attention_pointer
        raise ValueError(f"{self.decoder_kind} is not an ordered pointer decoder")

    def _recurrent_subset_decoder(self) -> RecurrentSubsetDecoder | None:
        if self.decoder_kind in ("lstm_pointer", "lstm_subset"):
            return self.lstm_subset
        if self.decoder_kind in ("gru_pointer", "gru_subset"):
            return self.gru_subset
        return None

    def _subset_prev_placeholder(self) -> torch.Tensor:
        recurrent = self._recurrent_subset_decoder()
        if recurrent is not None:
            return recurrent.prev_placeholder
        return self.attention_subset.prev_placeholder

    @staticmethod
    def _subset_context(
        selected_mask: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = selected_mask.size(1)
        selected_ratio = selected_mask.float().sum(dim=1) / max(num_nodes, 1)
        available_ratio = available.float().sum(dim=1) / max(num_nodes, 1)
        return torch.stack([selected_ratio, available_ratio], dim=1)

    def _subset_action_mask(
        self,
        *,
        batch: dict[str, torch.Tensor],
        problem: ProblemType,
        available: torch.Tensor,
        finished: torch.Tensor,
        selected_mask: torch.Tensor,
        remaining_capacity: torch.Tensor | None,
        covered_edges: torch.Tensor | None,
        current_loc: torch.Tensor | None,
        path_length: torch.Tensor | None,
    ) -> torch.Tensor:
        del selected_mask
        node_available = available.clone()
        if problem == "knapsack":
            if remaining_capacity is None:
                raise RuntimeError("knapsack decoder is missing remaining capacity")
            weights = self._require_tensor(batch, "weights").float()
            node_available &= weights <= remaining_capacity.view(-1, 1)
        if problem == "orienteering":
            if current_loc is None or path_length is None:
                raise RuntimeError("orienteering decoder is missing route state")
            coordinates = self._require_tensor(batch, "coordinates").float()
            depot = self._require_tensor(batch, "depot").float()
            budget = self._require_tensor(batch, "travel_budget").float()
            step = (coordinates - current_loc.unsqueeze(1)).norm(p=2, dim=-1)
            return_home = (coordinates - depot.unsqueeze(1)).norm(p=2, dim=-1)
            feasible_length = path_length.view(-1, 1) + step + return_home
            node_available &= feasible_length <= budget.view(-1, 1) + 1e-12

        stop_mask = torch.zeros(
            available.size(0),
            1,
            dtype=torch.bool,
            device=available.device,
        )
        if problem == "minimum_vertex_cover":
            adjacency = self._require_tensor(batch, "adjacency").bool()
            if covered_edges is None:
                raise RuntimeError("vertex cover decoder is missing covered edges")
            uncovered = adjacency & ~covered_edges
            stop_mask = uncovered.any(dim=(1, 2), keepdim=False).unsqueeze(1)

        action_mask = torch.cat([~node_available, stop_mask], dim=1)
        stop_only = torch.ones_like(action_mask)
        stop_only[:, -1] = False
        return torch.where(finished.unsqueeze(1), stop_only, action_mask)

    def _update_available(
        self,
        *,
        batch: dict[str, torch.Tensor],
        problem: ProblemType,
        available: torch.Tensor,
        active: torch.Tensor,
        node_selected: torch.Tensor,
        selected_mask: torch.Tensor,
        remaining_capacity: torch.Tensor | None,
        adjacency: torch.Tensor | None,
        covered_edges: torch.Tensor | None,
        current_loc: torch.Tensor | None,
        path_length: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_idx = torch.arange(available.size(0), device=available.device)
        active_idx = batch_idx[active]
        active_nodes = node_selected[active]
        available[active_idx, active_nodes] = False

        if problem == "mis":
            if adjacency is None:
                raise RuntimeError("MIS decoder is missing adjacency")
            available[active] &= ~adjacency[active_idx, active_nodes]
        elif problem == "maximum_clique":
            if adjacency is None:
                raise RuntimeError("clique decoder is missing adjacency")
            available[active] &= adjacency[active_idx, active_nodes]
            available &= ~selected_mask
        elif problem == "knapsack":
            if remaining_capacity is None:
                raise RuntimeError("knapsack decoder is missing remaining capacity")
            weights = self._require_tensor(batch, "weights").float()
            remaining_capacity[active] -= weights[active_idx, active_nodes]
        elif problem == "minimum_vertex_cover":
            if covered_edges is None:
                raise RuntimeError("vertex cover decoder is missing covered edges")
            covered_edges[active_idx, active_nodes, :] = True
            covered_edges[active_idx, :, active_nodes] = True
        elif problem == "orienteering":
            if current_loc is None or path_length is None:
                raise RuntimeError("orienteering decoder is missing route state")
            coordinates = self._require_tensor(batch, "coordinates").float()
            next_loc = coordinates[active_idx, active_nodes]
            path_length[active] += (next_loc - current_loc[active]).norm(p=2, dim=-1)
            current_loc[active] = next_loc
        return available

    def _decode_sigmoid_solution(
        self,
        batch: dict[str, torch.Tensor],
        logits: torch.Tensor,
        problem: ProblemType,
    ) -> torch.Tensor:
        if problem == "mis":
            return self._greedy_independent_set(batch, logits)
        if problem == "maximum_clique":
            return self._greedy_clique(batch, logits)
        if problem == "minimum_vertex_cover":
            return self._greedy_vertex_cover(batch, logits)
        if problem == "knapsack":
            return self._greedy_knapsack(batch, logits)
        if problem == "orienteering":
            return self._greedy_orienteering(batch, logits)
        raise ValueError(f"sigmoid_subset does not support {problem}")

    def _greedy_independent_set(
        self,
        batch: dict[str, torch.Tensor],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        adjacency = self._require_tensor(batch, "adjacency").bool()
        selected = torch.zeros_like(logits, dtype=torch.bool)
        for b in range(logits.size(0)):
            available = torch.ones(
                logits.size(1), dtype=torch.bool, device=logits.device
            )
            for node in logits[b].argsort(descending=True).tolist():
                score = float(logits[b, node].detach())
                if bool(available[node]) and score > 0.0:
                    selected[b, node] = True
                    available &= ~adjacency[b, node]
                    available[node] = False
        return selected

    def _greedy_clique(
        self,
        batch: dict[str, torch.Tensor],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        adjacency = self._require_tensor(batch, "adjacency").bool()
        selected = torch.zeros_like(logits, dtype=torch.bool)
        for b in range(logits.size(0)):
            available = torch.ones(
                logits.size(1), dtype=torch.bool, device=logits.device
            )
            for node in logits[b].argsort(descending=True).tolist():
                score = float(logits[b, node].detach())
                if bool(available[node]) and score > 0.0:
                    selected[b, node] = True
                    available &= adjacency[b, node]
                    available &= ~selected[b]
        return selected

    def _greedy_vertex_cover(
        self,
        batch: dict[str, torch.Tensor],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        adjacency = self._require_tensor(batch, "adjacency").bool()
        selected = logits > 0
        for b in range(logits.size(0)):
            for u in range(logits.size(1)):
                for v in range(u + 1, logits.size(1)):
                    if bool(adjacency[b, u, v]) and not (
                        bool(selected[b, u]) or bool(selected[b, v])
                    ):
                        pick = u if logits[b, u] >= logits[b, v] else v
                        selected[b, pick] = True
        return selected

    def _greedy_knapsack(
        self,
        batch: dict[str, torch.Tensor],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        weights = self._require_tensor(batch, "weights").float()
        capacity = self._require_tensor(batch, "capacity").float()
        selected = torch.zeros_like(logits, dtype=torch.bool)
        for b in range(logits.size(0)):
            remaining = float(capacity[b])
            for node in logits[b].argsort(descending=True).tolist():
                weight = float(weights[b, node])
                score = float(logits[b, node].detach())
                if weight <= remaining and score > 0.0:
                    selected[b, node] = True
                    remaining -= weight
        return selected

    def _greedy_orienteering(
        self,
        batch: dict[str, torch.Tensor],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        coordinates = self._require_tensor(batch, "coordinates").float()
        depot = self._require_tensor(batch, "depot").float()
        budget = self._require_tensor(batch, "travel_budget").float()
        selected = torch.zeros_like(logits, dtype=torch.bool)
        for b in range(logits.size(0)):
            current = depot[b]
            path_length = 0.0
            for node in logits[b].argsort(descending=True).tolist():
                score = float(logits[b, node].detach())
                step = float((coordinates[b, node] - current).norm(p=2))
                return_home = float((coordinates[b, node] - depot[b]).norm(p=2))
                if (
                    score > 0.0
                    and path_length + step + return_home <= float(budget[b]) + 1e-12
                ):
                    selected[b, node] = True
                    path_length += step
                    current = coordinates[b, node]
        return selected

    def _target_sequence(
        self,
        batch: dict[str, torch.Tensor],
        target_set: torch.Tensor,
    ) -> torch.Tensor:
        provided = batch.get("target_sequence")
        if isinstance(provided, torch.Tensor):
            return provided.long()
        stop_index = target_set.size(1)
        target = torch.full(
            (target_set.size(0), target_set.size(1) + 1),
            stop_index,
            dtype=torch.long,
            device=target_set.device,
        )
        for row in range(target_set.size(0)):
            selected = torch.nonzero(target_set[row] > 0.5, as_tuple=False).flatten()
            if selected.numel():
                target[row, : selected.numel()] = selected.long()
        return target

    @staticmethod
    def _static_order_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        selected_mask = torch.zeros_like(logits, dtype=torch.bool)
        losses: list[torch.Tensor] = []
        for step in range(target.size(1)):
            step_logits = logits.masked_fill(selected_mask, float("-inf"))
            step_target = target[:, step]
            losses.append(
                F.cross_entropy(
                    step_logits,
                    step_target,
                    reduction="none",
                    label_smoothing=label_smoothing,
                )
            )
            selected_mask = selected_mask.scatter(1, step_target.unsqueeze(1), True)
        return torch.stack(losses, dim=1).mean()

    def _cvrp_cost(
        self,
        batch: dict[str, torch.Tensor],
        order: torch.Tensor,
    ) -> torch.Tensor:
        depot = self._require_tensor(batch, "depot").float()
        coordinates = self._require_tensor(batch, "coordinates").float()
        demands = self._require_tensor(batch, "demands").float()
        capacity = self._require_tensor(batch, "capacity").float()
        values: list[torch.Tensor] = []
        for b in range(order.size(0)):
            total = torch.zeros((), dtype=coordinates.dtype, device=coordinates.device)
            current = depot[b]
            load = torch.zeros((), dtype=coordinates.dtype, device=coordinates.device)
            for node_tensor in order[b]:
                node = int(node_tensor.item())
                demand = demands[b, node]
                if bool(load > 0) and bool(load + demand > capacity[b]):
                    total = total + (current - depot[b]).norm(p=2)
                    current = depot[b]
                    load = torch.zeros(
                        (),
                        dtype=coordinates.dtype,
                        device=coordinates.device,
                    )
                total = total + (coordinates[b, node] - current).norm(p=2)
                current = coordinates[b, node]
                load = load + demand
            total = total + (current - depot[b]).norm(p=2)
            values.append(total)
        return torch.stack(values)

    def _resolve_problem(
        self,
        batch: dict[str, torch.Tensor],
        problem: ProblemType | None,
    ) -> ProblemType:
        if problem is not None:
            return problem
        if self.default_problem is not None:
            return self.default_problem
        if "adjacency" in batch:
            return "mis"
        if "weights" in batch:
            return "knapsack"
        if "prizes" in batch:
            return "orienteering"
        if "demands" in batch:
            return "cvrp"
        if "loc" in batch:
            return "tsp"
        raise ValueError("Cannot infer problem from batch")

    @staticmethod
    def _require_tensor(batch: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        value = batch.get(key)
        if value is None:
            raise ValueError(f"Missing batch['{key}']")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch['{key}'] must be a torch.Tensor")
        return value
