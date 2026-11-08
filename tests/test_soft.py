import copy

import pytest
import torch
from transformers import PreTrainedTokenizerBase, Qwen3_5TextModel

from rein.model import constants as mc
from rein.model.batch import collate
from rein.model.encode import RequestEncoder
from rein.model.prefix import shared_state_forward
from rein.model.soft import SoftPrompt, add_soft_tokens
from rein.schema import DecisionRequest
from tests.devices import DEVICE, HIDDEN_ATOL

FRONT = 3
TAIL = 2
BASIS_RANK = 4


@pytest.fixture
def soft_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    return copy.deepcopy(tokenizer)


@pytest.fixture
def soft_chat_tokenizer(chat_tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    return copy.deepcopy(chat_tokenizer)


def test_soft_tokens_are_new_special_tokens_in_the_spare_rows(soft_tokenizer: PreTrainedTokenizerBase) -> None:
    before = len(soft_tokenizer)
    front, tail = add_soft_tokens(soft_tokenizer, FRONT, TAIL)
    again = add_soft_tokens(soft_tokenizer, FRONT, TAIL)
    assert (front, tail) == again
    assert len(front) == FRONT and len(tail) == TAIL
    assert min(front + tail) >= before
    assert len(soft_tokenizer) == before + FRONT + TAIL
    text = mc.SOFT_TOKEN_FORMAT.format(index=0) + "A. foo" + mc.SOFT_TOKEN_FORMAT.format(index=FRONT)
    ids = soft_tokenizer(text, add_special_tokens=False)["input_ids"]
    assert ids[0] == front[0] and ids[-1] == tail[0]


def test_lettered_rows_carry_the_soft_tokens_around_the_question(
    soft_tokenizer: PreTrainedTokenizerBase, roleplay_request: DecisionRequest
) -> None:
    front, tail = add_soft_tokens(soft_tokenizer, FRONT, TAIL)
    encoder = RequestEncoder(soft_tokenizer, layout=mc.LAYOUT_LABELS, soft_front=front, soft_tail=tail)
    for question in encoder.encode(roleplay_request).questions:
        (row,) = question.rows
        assert row[1 : 1 + FRONT] == front
        assert row[-1 - TAIL : -1] == tail
        assert row[-1] == encoder.ids.query_mark


def test_soft_prompt_serves_its_rows_and_nothing_else(
    tiny_backbone: Qwen3_5TextModel, soft_tokenizer: PreTrainedTokenizerBase, roleplay_request: DecisionRequest
) -> None:
    front, tail = add_soft_tokens(soft_tokenizer, FRONT, TAIL)
    embedding = tiny_backbone.get_input_embeddings()
    plain = copy.deepcopy(embedding)
    soft = SoftPrompt(front + tail, embedding).to(DEVICE)
    with torch.no_grad():
        soft.weight.add_(1.0)
    handle = soft.attach(embedding)
    encoder = RequestEncoder(soft_tokenizer, layout=mc.LAYOUT_LABELS, soft_front=front, soft_tail=tail)
    row = torch.tensor([encoder.encode(roleplay_request).questions[0].rows[0]], device=DEVICE)
    served = embedding(row)
    reference = plain(row)
    is_soft = torch.isin(row, torch.tensor(front + tail, device=DEVICE))
    torch.testing.assert_close(served[~is_soft], reference[~is_soft])
    assert not torch.allclose(served[is_soft], reference[is_soft])
    served.sum().backward()
    assert soft.weight.grad is not None and bool((soft.weight.grad != 0).any())
    handle.remove()
    soft.merge_into(embedding)
    torch.testing.assert_close(embedding(row).detach(), served.detach(), atol=HIDDEN_ATOL, rtol=0)


def test_a_basis_keeps_the_served_rows_and_their_gradient_off_its_directions(
    tiny_backbone: Qwen3_5TextModel, soft_chat_tokenizer: PreTrainedTokenizerBase, roleplay_request: DecisionRequest
) -> None:
    front, tail = add_soft_tokens(soft_chat_tokenizer, FRONT, TAIL)
    embedding = tiny_backbone.get_input_embeddings()
    soft = SoftPrompt(front + tail, embedding).to(DEVICE)
    basis = torch.linalg.qr(torch.randn(embedding.embedding_dim, BASIS_RANK, device=DEVICE))[0].T
    soft.set_basis(basis)
    torch.testing.assert_close(soft.rows @ basis.T, torch.zeros(FRONT + TAIL, BASIS_RANK, device=DEVICE))
    torch.testing.assert_close(soft.rows, soft.weight.detach())
    encoder = RequestEncoder(soft_chat_tokenizer, layout=mc.LAYOUT_LABELS, soft_front=front, soft_tail=tail)
    row = torch.tensor([encoder.encode(roleplay_request).questions[0].rows[0]], device=DEVICE)
    soft.attach(embedding)
    served = embedding(row)
    (served * torch.randn_like(served)).sum().backward()
    assert soft.weight.grad is not None
    torch.testing.assert_close(soft.weight.grad @ basis.T, torch.zeros(FRONT + TAIL, BASIS_RANK, device=DEVICE))
    soft.merge_into(embedding)
    torch.testing.assert_close(embedding.weight[soft.ids].detach(), soft.rows.to(embedding.weight.dtype))


def test_soft_prompt_changes_the_rows_but_not_the_state(
    tiny_backbone: Qwen3_5TextModel, soft_tokenizer: PreTrainedTokenizerBase, roleplay_request: DecisionRequest
) -> None:
    front, tail = add_soft_tokens(soft_tokenizer, FRONT, TAIL)
    encoder = RequestEncoder(soft_tokenizer, layout=mc.LAYOUT_LABELS, soft_front=front, soft_tail=tail)
    batch = collate([encoder.encode(roleplay_request)], encoder.pad_id).to(DEVICE)
    with torch.no_grad():
        before = shared_state_forward(tiny_backbone, batch)
        soft = SoftPrompt(front + tail, tiny_backbone.get_input_embeddings()).to(DEVICE)
        soft.weight.add_(1.0)
        soft.attach(tiny_backbone.get_input_embeddings())
        after = shared_state_forward(tiny_backbone, batch)
    assert not torch.allclose(before, after, atol=HIDDEN_ATOL)
