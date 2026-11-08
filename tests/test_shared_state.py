import torch
from transformers import Qwen3_5TextModel

from rein.model.batch import collate
from rein.model.encode import EncodedRequest, RequestEncoder
from rein.model.prefix import shared_state_forward
from rein.schema import DecisionRequest
from tests.devices import (
    DEVICE,
    GRAD_ELEMENT_ATOL,
    GRAD_GLOBAL_FLOOR,
    GRAD_GLOBAL_RELATIVE_L2,
    GRAD_TENSOR_RELATIVE_L2,
    HIDDEN_ATOL,
    ON_GPU,
)

GRAD_RTOL = 1e-2
WEIGHT_SEED = 7


def _relative_l2(grads_a: tuple[torch.Tensor | None, ...], grads_b: tuple[torch.Tensor | None, ...]) -> float:
    pairs = [(a, b) for a, b in zip(grads_a, grads_b, strict=True) if a is not None and b is not None]
    diff = torch.sqrt(sum(((a - b) ** 2).sum() for a, b in pairs))
    norm = torch.sqrt(sum((b**2).sum() for _, b in pairs))
    return float(diff / norm)


def _encoded(encoder: RequestEncoder, *requests: DecisionRequest) -> list[EncodedRequest]:
    return [encoder.encode(request) for request in requests]


def _row_tokens(encoded: list[EncodedRequest]) -> list[tuple[int, tuple[int, ...]]]:
    return [
        (owner, row) for owner, request in enumerate(encoded) for question in request.questions for row in question.rows
    ]


def _full_hidden(backbone: Qwen3_5TextModel, state: tuple[int, ...], row: tuple[int, ...]) -> torch.Tensor:
    ids = torch.tensor([list(state) + list(row)], device=DEVICE)
    return backbone(input_ids=ids, use_cache=False).last_hidden_state[0, len(state) :]


def test_rows_over_a_shared_state_match_full_sequences(
    tiny_backbone: Qwen3_5TextModel,
    encoder: RequestEncoder,
    roleplay_request: DecisionRequest,
    tool_call_request: DecisionRequest,
) -> None:
    encoded = _encoded(encoder, roleplay_request, tool_call_request)
    assert len(encoded[0].state_ids) != len(encoded[1].state_ids)
    batch = collate(encoded, encoder.pad_id).to(DEVICE)
    with torch.no_grad():
        shared = shared_state_forward(tiny_backbone, batch)
        for index, (owner, row) in enumerate(_row_tokens(encoded)):
            reference = _full_hidden(tiny_backbone, encoded[owner].state_ids, row)
            torch.testing.assert_close(shared[index, : len(row)], reference, atol=HIDDEN_ATOL, rtol=0)


def test_gradients_through_the_shared_state_match_full_sequences(
    tiny_backbone: Qwen3_5TextModel,
    encoder: RequestEncoder,
    roleplay_request: DecisionRequest,
    tool_call_request: DecisionRequest,
) -> None:
    encoded = _encoded(encoder, roleplay_request, tool_call_request)
    batch = collate(encoded, encoder.pad_id).to(DEVICE)
    rows = _row_tokens(encoded)
    generator = torch.Generator().manual_seed(WEIGHT_SEED)
    weights = [
        torch.randn(len(row), tiny_backbone.config.hidden_size, generator=generator).to(DEVICE) for _, row in rows
    ]
    named = [(name, param) for name, param in tiny_backbone.named_parameters() if param.requires_grad]
    params = [param for _, param in named]

    shared = shared_state_forward(tiny_backbone, batch)
    shared_loss = sum((shared[i, : len(row)] * weights[i]).sum() for i, (_, row) in enumerate(rows))
    shared_grads = torch.autograd.grad(shared_loss, params, allow_unused=True)

    full_loss = sum(
        (_full_hidden(tiny_backbone, encoded[owner].state_ids, row) * weights[i]).sum()
        for i, (owner, row) in enumerate(rows)
    )
    full_grads = torch.autograd.grad(full_loss, params, allow_unused=True)

    assert _relative_l2(shared_grads, full_grads) < GRAD_GLOBAL_RELATIVE_L2
    global_norm = float(torch.sqrt(sum((grad**2).sum() for grad in full_grads if grad is not None)))
    for (name, _), shared_grad, full_grad in zip(named, shared_grads, full_grads, strict=True):
        if full_grad is None:
            assert shared_grad is None or torch.count_nonzero(shared_grad) == 0
            continue
        assert shared_grad is not None
        if ON_GPU:
            error = _relative_l2((shared_grad,), (full_grad,))
            reference = float(full_grad.norm())
            difference = float((shared_grad - full_grad).norm())
            tolerance = max(GRAD_TENSOR_RELATIVE_L2 * reference, GRAD_GLOBAL_FLOOR * global_norm)
            assert difference <= tolerance, (
                f"{name}: relative L2 {error:.3g}, |diff| {difference:.3g}, |ref| {reference:.3g}, "
                f"|global ref| {global_norm:.3g}"
            )
        else:
            torch.testing.assert_close(shared_grad, full_grad, atol=GRAD_ELEMENT_ATOL, rtol=GRAD_RTOL)


def test_checkpointing_keeps_outputs_and_gradients(
    tiny_backbone: Qwen3_5TextModel, encoder: RequestEncoder, roleplay_request: DecisionRequest
) -> None:
    batch = collate(_encoded(encoder, roleplay_request), encoder.pad_id).to(DEVICE)
    params = [param for param in tiny_backbone.parameters() if param.requires_grad]
    tiny_backbone.train()
    plain = shared_state_forward(tiny_backbone, batch, checkpointing=False)
    plain_grads = torch.autograd.grad(plain.square().sum(), params, allow_unused=True)
    checkpointed = shared_state_forward(tiny_backbone, batch, checkpointing=True)
    checkpointed_grads = torch.autograd.grad(checkpointed.square().sum(), params, allow_unused=True)
    torch.testing.assert_close(checkpointed, plain, atol=HIDDEN_ATOL, rtol=0)
    for a, b in zip(plain_grads, checkpointed_grads, strict=True):
        if a is not None and b is not None:
            torch.testing.assert_close(a, b, atol=GRAD_ELEMENT_ATOL, rtol=GRAD_RTOL)


def test_middle_layer_readout_matches_a_truncated_reference(
    tiny_backbone: Qwen3_5TextModel, encoder: RequestEncoder, roleplay_request: DecisionRequest
) -> None:
    from rein.model.backbone import truncate_backbone

    truncated = truncate_backbone(tiny_backbone, 2)
    assert len(truncated.layers) == 2
    encoded = _encoded(encoder, roleplay_request)
    batch = collate(encoded, encoder.pad_id).to(DEVICE)
    with torch.no_grad():
        shared = shared_state_forward(truncated, batch)
        for index, (owner, row) in enumerate(_row_tokens(encoded)):
            reference = _full_hidden(truncated, encoded[owner].state_ids, row)
            torch.testing.assert_close(shared[index, : len(row)], reference, atol=HIDDEN_ATOL, rtol=0)
