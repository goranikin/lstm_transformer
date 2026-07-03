import math

import torch
import torch.nn.functional as F
from torch import nn

from src_new.constants import DecodeType
from src_new.core import EncoderOutput, ProblemDecodeState, SolutionOutput


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(mask, float("-inf"))
    log_p = F.log_softmax(masked, dim=-1)
    return torch.nan_to_num(log_p, nan=0.0, neginf=0.0)


def select_action(
    logits: torch.Tensor,
    mask: torch.Tensor,
    decode_type: DecodeType,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked = logits.masked_fill(mask, float("-inf"))
    if decode_type == "greedy":
        action = masked.argmax(dim=-1)
    else:
        probs = torch.softmax(masked, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        action = torch.multinomial(probs, 1).squeeze(1)
    log_p = masked_log_softmax(logits, mask).gather(1, action.unsqueeze(1)).squeeze(1)
    return action, log_p


class AutoregressiveDecoder(nn.Module):
    def decode(
        self,
        encoder_output: EncoderOutput,
        problem_state: ProblemDecodeState,
        decode_type: DecodeType,
    ) -> SolutionOutput:
        problem = problem_state.problem
        state = problem_state.state
        target_actions = problem_state.target_actions
        log_probs: list[torch.Tensor] = []
        max_steps = max(1, state.selected_mask.size(1) * 2 + 2)

        for step in range(max_steps):
            done_before = problem.is_done(state)
            if bool(done_before.all()):
                break
            mask = problem.get_mask(state)
            logits = self.step_logits(encoder_output, problem_state, mask)
            if target_actions is not None and step < target_actions.size(1):
                fallback, _ = select_action(logits, mask, "greedy")
                raw_action = target_actions[:, step].to(logits.device)
                action = torch.where(raw_action >= 0, raw_action.long(), fallback)
                log_p = masked_log_softmax(logits, mask).gather(
                    1,
                    action.unsqueeze(1),
                ).squeeze(1)
            else:
                action, log_p = select_action(logits, mask, decode_type)
            log_p = torch.where(done_before, torch.zeros_like(log_p), log_p)
            state = problem.step(state, action)
            problem_state.state = state
            log_probs.append(log_p)

        solution = problem.to_solution(state)
        objective = problem.compute_objective(problem_state.batch, solution)
        feasible = problem.check_feasible(problem_state.batch, solution)
        log_prob_tensor = (
            torch.stack(log_probs, dim=1).sum(dim=1)
            if log_probs
            else torch.zeros_like(objective)
        )
        return SolutionOutput(
            actions=solution["actions"],
            log_probs=log_prob_tensor,
            selected_mask=solution.get("selected_mask"),
            objective=objective,
            feasible=feasible,
            reward=problem.reward(objective),
        )

    def step_logits(
        self,
        encoder_output: EncoderOutput,
        problem_state: ProblemDecodeState,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _action_embeddings(
        node_embeddings: torch.Tensor,
        action: torch.Tensor,
        placeholder: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, d_model = node_embeddings.shape
        safe = action.clamp(min=0, max=node_count - 1)
        gathered = node_embeddings.gather(
            1,
            safe.view(batch_size, 1, 1).expand(-1, 1, d_model),
        ).squeeze(1)
        return torch.where((action >= 0).unsqueeze(1), gathered, placeholder.expand(batch_size, -1))

    @staticmethod
    def _append_stop_key(
        keys: torch.Tensor,
        stop_key: torch.Tensor,
        action_count: int,
    ) -> torch.Tensor:
        if action_count == keys.size(1):
            return keys
        if action_count != keys.size(1) + 1:
            raise ValueError("Action mask size must be node_count or node_count + 1")
        return torch.cat([keys, stop_key.view(1, 1, -1).expand(keys.size(0), 1, -1)], dim=1)


class AttentionPointerDecoder(AutoregressiveDecoder):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        tanh_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.tanh_clip = tanh_clip
        self.query_proj = nn.Linear(3 * d_model + context_dim, d_model)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.glimpse_proj = nn.Linear(d_model, d_model)
        self.prev_placeholder = nn.Parameter(torch.empty(d_model))
        self.first_placeholder = nn.Parameter(torch.empty(d_model))
        self.stop_key = nn.Parameter(torch.empty(d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.query_proj, self.key_proj, self.value_proj, self.glimpse_proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.prev_placeholder, std=0.02)
        nn.init.normal_(self.first_placeholder, std=0.02)
        nn.init.normal_(self.stop_key, std=0.02)

    def step_logits(
        self,
        encoder_output: EncoderOutput,
        problem_state: ProblemDecodeState,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        h = encoder_output.node_embeddings
        state = problem_state.state
        prev = self._action_embeddings(h, state.prev_action, self.prev_placeholder)
        first = self._action_embeddings(h, state.first_action, self.first_placeholder)
        context = problem_state.problem.context_features(state).to(h.dtype)
        query = self.query_proj(
            torch.cat([encoder_output.graph_embedding, prev, first, context], dim=-1)
        )
        keys = self.key_proj(h)
        values = self.value_proj(h)
        node_mask = mask[:, : h.size(1)]
        attn_logits = torch.matmul(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
        attn_logits = attn_logits / math.sqrt(self.d_model)
        attn = torch.softmax(attn_logits.masked_fill(node_mask, float("-inf")), dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        glimpse = self.glimpse_proj(torch.bmm(attn.unsqueeze(1), values).squeeze(1))
        logit_keys = self._append_stop_key(keys, self.stop_key, mask.size(1))
        logits = torch.matmul(glimpse.unsqueeze(1), logit_keys.transpose(1, 2)).squeeze(1)
        logits = logits / math.sqrt(self.d_model)
        if self.tanh_clip > 0:
            logits = self.tanh_clip * torch.tanh(logits)
        return logits


class RecurrentPointerDecoder(AutoregressiveDecoder):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        cell_kind: str,
        tanh_clip: float = 10.0,
    ) -> None:
        super().__init__()
        if cell_kind not in {"lstm", "gru"}:
            raise ValueError("cell_kind must be 'lstm' or 'gru'")
        self.d_model = d_model
        self.context_dim = context_dim
        self.cell_kind = cell_kind
        self.tanh_clip = tanh_clip
        self.input_proj = nn.Linear(d_model + context_dim, d_model)
        self.hidden_init = nn.Linear(d_model, d_model)
        self.cell_init = nn.Linear(d_model, d_model) if cell_kind == "lstm" else None
        self.cell = (
            nn.LSTMCell(d_model, d_model)
            if cell_kind == "lstm"
            else nn.GRUCell(d_model, d_model)
        )
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.prev_placeholder = nn.Parameter(torch.empty(d_model))
        self.stop_key = nn.Parameter(torch.empty(d_model))
        self._hidden: torch.Tensor | None = None
        self._cell_state: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.input_proj, self.hidden_init, self.key_proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if self.cell_init is not None:
            nn.init.xavier_uniform_(self.cell_init.weight)
            nn.init.zeros_(self.cell_init.bias)
        nn.init.normal_(self.prev_placeholder, std=0.02)
        nn.init.normal_(self.stop_key, std=0.02)

    def decode(
        self,
        encoder_output: EncoderOutput,
        problem_state: ProblemDecodeState,
        decode_type: DecodeType,
    ) -> SolutionOutput:
        self._hidden = torch.tanh(self.hidden_init(encoder_output.graph_embedding))
        self._cell_state = (
            torch.tanh(self.cell_init(encoder_output.graph_embedding))
            if self.cell_init is not None
            else None
        )
        try:
            return super().decode(encoder_output, problem_state, decode_type)
        finally:
            self._hidden = None
            self._cell_state = None

    def step_logits(
        self,
        encoder_output: EncoderOutput,
        problem_state: ProblemDecodeState,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self._hidden is None:
            raise RuntimeError("Recurrent decoder state is not initialized")
        h = encoder_output.node_embeddings
        state = problem_state.state
        prev = self._action_embeddings(h, state.prev_action, self.prev_placeholder)
        context = problem_state.problem.context_features(state).to(h.dtype)
        decoder_input = self.input_proj(torch.cat([prev, context], dim=-1))
        if self.cell_kind == "lstm":
            if self._cell_state is None or not isinstance(self.cell, nn.LSTMCell):
                raise RuntimeError("LSTM decoder state is not initialized")
            self._hidden, self._cell_state = self.cell(
                decoder_input,
                (self._hidden, self._cell_state),
            )
        else:
            if not isinstance(self.cell, nn.GRUCell):
                raise RuntimeError("GRU decoder is misconfigured")
            self._hidden = self.cell(decoder_input, self._hidden)
        keys = self._append_stop_key(self.key_proj(h), self.stop_key, mask.size(1))
        logits = torch.matmul(self._hidden.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
        logits = logits / math.sqrt(self.d_model)
        if self.tanh_clip > 0:
            logits = self.tanh_clip * torch.tanh(logits)
        return logits


class LSTMPointerDecoder(RecurrentPointerDecoder):
    def __init__(self, d_model: int, context_dim: int, tanh_clip: float = 10.0) -> None:
        super().__init__(d_model, context_dim, cell_kind="lstm", tanh_clip=tanh_clip)


class GRUPointerDecoder(RecurrentPointerDecoder):
    def __init__(self, d_model: int, context_dim: int, tanh_clip: float = 10.0) -> None:
        super().__init__(d_model, context_dim, cell_kind="gru", tanh_clip=tanh_clip)


class SigmoidSubsetDecoder(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(d_model, 1)

    def logits(self, encoder_output: EncoderOutput) -> torch.Tensor:
        return self.classifier(encoder_output.node_embeddings).squeeze(-1)

    def decode(
        self,
        encoder_output: EncoderOutput,
        problem_state: ProblemDecodeState,
        decode_type: DecodeType,
    ) -> SolutionOutput:
        logits = self.logits(encoder_output)
        log_probs = None
        proposed_mask = None
        if decode_type == "sampling":
            probs = torch.sigmoid(logits)
            proposed_mask = torch.bernoulli(probs).bool()
            log_probs = (
                proposed_mask.float() * F.logsigmoid(logits)
                + (~proposed_mask).float() * F.logsigmoid(-logits)
            ).sum(dim=1)
        elif decode_type == "greedy":
            proposed_mask = logits > 0
        else:
            raise ValueError("decode_type must be 'greedy' or 'sampling'")

        solution = problem_state.problem.repair_solution(
            problem_state.batch,
            logits,
            proposed_mask,
        )
        objective = problem_state.problem.compute_objective(problem_state.batch, solution)
        feasible = problem_state.problem.check_feasible(problem_state.batch, solution)
        return SolutionOutput(
            actions=solution["actions"],
            log_probs=log_probs,
            selected_mask=solution.get("selected_mask"),
            objective=objective,
            feasible=feasible,
            logits=logits,
            reward=problem_state.problem.reward(objective),
        )
