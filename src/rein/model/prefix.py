from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextModel

from rein.model import constants as c
from rein.model.batch import DecisionBatch

if TYPE_CHECKING:
    from peft.tuners.tuners_utils import BaseTunerLayer


class LayerPrefix:
    """Cache for one decoder layer: records the state pass, then serves it to the option rows.

    It duck-types the cache calls a Qwen3.5 decoder layer makes and never writes in place, so
    gradients flow from every row back into the shared state pass.
    """

    def __init__(self, owner: torch.Tensor) -> None:
        self._owner = owner
        self._serving = False
        self._conv_tail: torch.Tensor | None = None
        self._recurrent: torch.Tensor | None = None
        self._keys: torch.Tensor | None = None
        self._values: torch.Tensor | None = None

    def serve_rows(self) -> None:
        self._serving = True

    def get_conv_state(self, layer_idx: int, state_idx: int = 0) -> None:
        return None

    def get_recurrent_state(self, layer_idx: int, state_idx: int = 0) -> torch.Tensor | None:
        if not self._serving or self._recurrent is None:
            return None
        return self._recurrent.index_select(0, self._owner)

    def is_recording_past(self, layer_idx: int) -> bool:
        return False

    def update_conv_state(
        self, conv_states: torch.Tensor, layer_idx: int, conv_kernel_size: int | None = None, **kwargs: Any
    ) -> torch.Tensor:
        if not self._serving:
            kernel = conv_kernel_size or conv_states.shape[-1]
            tail = conv_states[..., -kernel:]
            if tail.shape[-1] < kernel:
                tail = F.pad(tail, (kernel - tail.shape[-1], 0))
            self._conv_tail = tail
            return conv_states
        if self._conv_tail is None:
            raise RuntimeError("rows ran before the state pass recorded the conv tail")
        tail = self._conv_tail.index_select(0, self._owner).to(conv_states.dtype)
        return torch.cat([tail, conv_states], dim=-1)

    def update_recurrent_state(self, recurrent_states: torch.Tensor, layer_idx: int, **kwargs: Any) -> torch.Tensor:
        if not self._serving:
            self._recurrent = recurrent_states
        return recurrent_states

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._serving:
            self._keys, self._values = key_states, value_states
            return key_states, value_states
        if self._keys is None or self._values is None:
            raise RuntimeError("rows ran before the state pass recorded keys and values")
        keys = torch.cat([self._keys.index_select(0, self._owner), key_states], dim=-2)
        values = torch.cat([self._values.index_select(0, self._owner), value_states], dim=-2)
        return keys, values


@dataclass(frozen=True)
class _Masks:
    full_state: torch.Tensor
    linear_state: torch.Tensor
    state_valid: torch.Tensor
    full_rows: torch.Tensor


def _attention_mask(allow: torch.Tensor, implementation: str, dtype: torch.dtype) -> torch.Tensor:
    if implementation == c.SDPA:
        return allow
    return torch.zeros(allow.shape, dtype=dtype, device=allow.device).masked_fill(~allow, torch.finfo(dtype).min)


def _build_masks(batch: DecisionBatch, implementation: str, dtype: torch.dtype) -> _Masks:
    device = batch.state_mask.device
    state_len = batch.state_mask.shape[1]
    row_len = batch.row_mask.shape[1]
    state_causal = torch.tril(torch.ones(state_len, state_len, dtype=torch.bool, device=device))
    state_eye = torch.eye(state_len, dtype=torch.bool, device=device)
    full_state = (state_causal[None] & batch.state_mask[:, None, :]) | state_eye[None]

    row_causal = torch.tril(torch.ones(row_len, row_len, dtype=torch.bool, device=device))
    row_eye = torch.eye(row_len, dtype=torch.bool, device=device)
    row_part = (row_causal[None] & batch.row_mask[:, None, :]) | row_eye[None]
    state_part = batch.state_mask.index_select(0, batch.row_owner)[:, None, :].expand(-1, row_len, -1)
    full_rows = torch.cat([state_part, row_part], dim=-1)

    return _Masks(
        full_state=_attention_mask(full_state[:, None], implementation, dtype),
        linear_state=batch.state_mask.to(dtype),
        state_valid=batch.state_mask,
        full_rows=_attention_mask(full_rows[:, None], implementation, dtype),
    )


def _freeze_padding_decay(valid: torch.Tensor, module: nn.Module, inputs: object, output: torch.Tensor) -> torch.Tensor:
    """Pushes the decay logit of left-padding tokens to -inf-like, so softplus gives g = 0 there.

    Padding then leaves the DeltaNet state and its gradients exactly untouched, also inside fused kernels that
    would otherwise leak a decay gradient from padded positions into A_log and dt_bias.
    """
    return torch.where(valid[..., None], output, torch.full_like(output, c.PAD_DECAY_LOGIT))


def adapters_in(layer: nn.Module) -> list["BaseTunerLayer"]:
    """The LoRA layers inside a block; none when peft is not installed, since then no adapter can be attached."""
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:
        return []
    return [module for module in layer.modules() if isinstance(module, BaseTunerLayer)]


def _layer_step(
    layer: Qwen3_5DecoderLayer,
    state_hidden: torch.Tensor,
    row_hidden: torch.Tensor,
    *,
    state_rope: tuple[torch.Tensor, torch.Tensor],
    row_rope: tuple[torch.Tensor, torch.Tensor],
    masks: _Masks,
    owner: torch.Tensor,
    paused_adapters: Sequence["BaseTunerLayer"] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    cache = LayerPrefix(owner)
    full = layer.block_type == c.FULL_ATTENTION
    handle = None
    if not full:
        handle = layer.linear_attn.in_proj_a.register_forward_hook(partial(_freeze_padding_decay, masks.state_valid))
    for adapter in paused_adapters:
        adapter.enable_adapters(False)
    try:
        state_out = layer(
            state_hidden,
            position_embeddings=state_rope,
            attention_mask=masks.full_state if full else masks.linear_state,
            past_key_values=cache,
        )
    finally:
        if handle is not None:
            handle.remove()
        for adapter in paused_adapters:
            adapter.enable_adapters(True)
    cache.serve_rows()
    row_out = layer(
        row_hidden,
        position_embeddings=row_rope,
        attention_mask=masks.full_rows if full else None,
        past_key_values=cache,
    )
    return state_out, row_out


LayerCapture = Callable[[int, torch.Tensor], None]


def shared_state_forward(
    backbone: Qwen3_5TextModel,
    batch: DecisionBatch,
    checkpointing: bool = False,
    capture: LayerCapture | None = None,
    adapter_on_state: bool = True,
) -> torch.Tensor:
    """Hidden states of every option row, each continuing from its record's state, encoded once.

    `capture(layer_number, row_hidden)` sees the rows after every block (1-based), for layer sweeps. With
    `adapter_on_state` off, the LoRA layers rest during the state pass and act on the rows only, as an activated LoRA
    does after its invocation tokens: the state is the base model's, and a runtime may reuse its cache.
    """
    embed = backbone.get_input_embeddings()
    state_hidden = embed(batch.state_ids)
    row_hidden = embed(batch.row_ids)
    state_positions = (batch.state_mask.long().cumsum(-1) - 1).clamp(min=0)
    state_lengths = batch.state_mask.long().sum(-1)
    row_offsets = torch.arange(batch.row_ids.shape[1], device=batch.row_ids.device)
    row_positions = state_lengths.index_select(0, batch.row_owner)[:, None] + row_offsets[None]
    state_rope = backbone.rotary_emb(state_hidden, state_positions)
    row_rope = backbone.rotary_emb(row_hidden, row_positions)
    masks = _build_masks(batch, backbone.config._attn_implementation, state_hidden.dtype)

    for number, layer in enumerate(backbone.layers, start=1):
        step = partial(
            _layer_step,
            cast(Qwen3_5DecoderLayer, layer),
            state_rope=state_rope,
            row_rope=row_rope,
            masks=masks,
            owner=batch.row_owner,
            paused_adapters=() if adapter_on_state else adapters_in(layer),
        )
        if checkpointing and torch.is_grad_enabled():
            state_hidden, row_hidden = checkpoint(step, state_hidden, row_hidden, use_reentrant=False)
        else:
            state_hidden, row_hidden = step(state_hidden, row_hidden)
        if capture is not None:
            capture(number, row_hidden)
    return backbone.norm(row_hidden)
