import pytest
import torch

from rein.model.head import LabelHead, SetPointerHead

LABEL_VOCAB = 50

HIDDEN = 24
QUESTIONS = 3
MAX_OPTIONS = 5
ATOL = 1e-5


def _inputs(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    h_query = torch.randn(QUESTIONS, HIDDEN, generator=generator)
    h_options = torch.randn(QUESTIONS, MAX_OPTIONS, HIDDEN, generator=generator)
    mask = torch.tensor([[True] * 5, [True, True, False, False, False], [True, True, True, False, False]])
    return h_query, h_options, mask


@pytest.mark.parametrize("set_block", [True, False])
def test_head_is_permutation_equivariant(set_block: bool) -> None:
    torch.manual_seed(0)
    head = SetPointerHead(hidden_size=HIDDEN, dim=16, heads=2, set_block=set_block).eval()
    h_query, h_options, mask = _inputs()
    base = head(h_query, h_options, mask)
    perm = torch.tensor([4, 2, 0, 3, 1])
    full_mask = mask[0:1]
    permuted = head(h_query[0:1], h_options[0:1, perm], full_mask)
    torch.testing.assert_close(permuted[0], base[0, perm], atol=ATOL, rtol=0)


def test_label_head_reads_the_logits_of_the_label_tokens_and_owns_no_weights() -> None:
    torch.manual_seed(0)
    embeddings = torch.nn.Embedding(LABEL_VOCAB, HIDDEN)
    head = LabelHead(embeddings)
    h_query, _, mask = _inputs()
    label_ids = torch.randint(0, LABEL_VOCAB, (QUESTIONS, MAX_OPTIONS))
    logits = head(h_query, label_ids, mask)
    expected = torch.einsum("qd,qod->qo", h_query, embeddings.weight[label_ids])
    torch.testing.assert_close(logits[mask], expected[mask], atol=ATOL, rtol=0)
    assert bool((logits[~mask] == torch.finfo(logits.dtype).min).all())
    assert list(head.parameters()) == []
    assert head.state_dict() == {}


def test_padded_options_do_not_leak_into_valid_logits() -> None:
    torch.manual_seed(0)
    head = SetPointerHead(hidden_size=HIDDEN, dim=16, heads=2).eval()
    h_query, h_options, mask = _inputs()
    base = head(h_query, h_options, mask)
    noisy = h_options.clone()
    noisy[~mask] = torch.randn(int((~mask).sum()), HIDDEN) * 50
    other = head(h_query, noisy, mask)
    torch.testing.assert_close(other[mask], base[mask], atol=ATOL, rtol=0)
    assert torch.all(other[~mask] == torch.finfo(other.dtype).min)
