import random

import pytest
import torch
from transformers import GenerationMixin, Qwen3_5TextModel

from rein.compiler.emit import firing_gate, rule_neurons
from rein.compiler.model import Compiler, decision_logits, null_option_index
from rein.compiler.reader import RuleReader
from rein.model.encode import RequestEncoder
from rein.neurons.units import logit, mlp_contribution
from rein.schema import Question
from tests.conftest import TINY_HIDDEN
from tests.devices import PROB_ATOL

BLOCKS = (1, 3)
DIM, HEADS, LAYERS, RANK = 16, 2, 2, 8
STATES = 64
TEMPERATURE = 1.3
GATE = 0.85
GAIN, WIDTH, UP_SCALE = 40.0, 4.0, 1.0
DECIDED_MARGIN = 0.5
ORDER_ATOL = 1e-5
BASE = "You keep the tavern's house rules."
CHOICE = Question(
    type="choice",
    instructions="Which tool does the latest request need?",
    criteria={"give_item": "hand over an item", "ring_bell": "ring the signal bell", "none": None},
)
NOUL = Question(type="noul", instructions="Is the visitor asking about real-world politics?")


@pytest.fixture
def reader(tiny_backbone: Qwen3_5TextModel, encoder: RequestEncoder) -> RuleReader:
    return RuleReader(tiny_backbone, encoder)


@pytest.fixture
def compiler() -> Compiler:
    torch.manual_seed(0)
    return Compiler(TINY_HIDDEN, BLOCKS, dim=DIM, heads=HEADS, layers=LAYERS, rank=RANK).eval()


def _states_on_the_channel(channel: torch.Tensor) -> torch.Tensor:
    """Random MLP inputs moved along the channel so that `channel . x` is exactly 1, block by block."""
    states = torch.randn(STATES, len(BLOCKS), TINY_HIDDEN, generator=torch.Generator().manual_seed(3))
    reading = torch.einsum("nbh,bh->nb", states, channel)
    return states + ((1.0 - reading) / channel.pow(2).sum(-1))[..., None] * channel


def test_4_compiled_neurons_follow_the_options_in_any_order(reader: RuleReader, compiler: Compiler) -> None:
    keys = list(CHOICE.criteria)
    for seed in (1, 2, 3):
        order = keys[:]
        random.Random(seed).shuffle(order)
        shuffled = CHOICE.model_copy(update={"criteria": {key: CHOICE.criteria[key] for key in order}})
        with torch.no_grad():
            base = compiler(reader.read([(CHOICE, BASE)]))
            moved = compiler(reader.read([(shuffled, BASE)]))
        permutation = [keys.index(key) for key in order]
        torch.testing.assert_close(moved.detectors[0], base.detectors[0, permutation], atol=ORDER_ATOL, rtol=0)
        torch.testing.assert_close(moved.offsets[0], base.offsets[0, permutation], atol=ORDER_ATOL, rtol=0)


def test_5_probabilities_of_each_rule_sum_to_one(reader: RuleReader, compiler: Compiler) -> None:
    with torch.no_grad():
        compiled = compiler(reader.read([(CHOICE, BASE), (NOUL, BASE)]))
    channel = torch.randn(len(BLOCKS), TINY_HIDDEN)
    logits = decision_logits(compiled, torch.randn(2, len(BLOCKS), TINY_HIDDEN), channel)
    probs = torch.softmax(logits, dim=-1).masked_fill(~compiled.mask[:, None, :], 0.0)
    torch.testing.assert_close(probs.sum(-1), torch.ones(2, len(BLOCKS)), atol=PROB_ATOL, rtol=0)
    assert torch.all(probs[1, :, 2:] == 0)


def test_3_the_compiled_neuron_decides_exactly_like_the_compiler(reader: RuleReader, compiler: Compiler) -> None:
    with torch.no_grad():
        compiled = compiler(reader.read([(NOUL, BASE)]))
    channel = torch.randn(len(BLOCKS), TINY_HIDDEN, generator=torch.Generator().manual_seed(4))
    states = _states_on_the_channel(channel)
    yes, no = NOUL.option_keys().index("yes"), null_option_index(NOUL.option_keys())
    for position, block in enumerate(BLOCKS):
        logits = decision_logits(compiled, states, channel)[:, position, :2]
        p_yes = torch.softmax(logits / TEMPERATURE, dim=-1)[:, yes]
        gate = firing_gate(compiled, 0, yes, no, block, channel[position], TEMPERATURE, GATE)
        pre_activation = states[:, position] @ gate
        expected = torch.logit(p_yes.double()).float() - logit(GATE)
        torch.testing.assert_close(pre_activation, expected, atol=1e-3, rtol=1e-4)

        steer = torch.nn.functional.one_hot(torch.tensor(0), TINY_HIDDEN).float()
        neurons = rule_neurons(gate, channel[position], steer, GAIN, WIDTH, UP_SCALE)
        written = mlp_contribution(states[:, position], neurons)[:, 0]
        decided = expected.abs() > DECIDED_MARGIN
        fires = p_yes >= GATE
        assert torch.all((written[decided] > WIDTH * UP_SCALE / 2) == fires[decided])


def test_compiling_never_generates_or_touches_an_lm_head(
    reader: RuleReader, compiler: Compiler, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("generate() must never run while compiling")

    monkeypatch.setattr(GenerationMixin, "generate", forbidden)
    with torch.no_grad():
        compiler(reader.read([(NOUL, BASE)]))
    names = [name for name, _ in reader.backbone.named_modules()] + [name for name, _ in compiler.named_modules()]
    assert not any("lm_head" in name for name in names)


def test_null_options_cover_yes_no_and_choice_rules() -> None:
    assert null_option_index(NOUL.option_keys()) == 0
    assert null_option_index(CHOICE.option_keys()) == 2
    score = Question(type="score", instructions="How far off tone?", criteria=["on", "off"])
    assert null_option_index(score.option_keys()) == -1


FACETS = 3


def test_each_facet_compiles_to_one_exact_neuron_against_the_null_option(reader: RuleReader) -> None:
    torch.manual_seed(1)
    faceted = Compiler(TINY_HIDDEN, BLOCKS, dim=DIM, heads=HEADS, layers=LAYERS, rank=RANK, facets=FACETS).eval()
    with torch.no_grad():
        compiled = faceted(reader.read([(NOUL, BASE)]))
    assert tuple(compiled.detectors.shape[2:4]) == (len(BLOCKS), FACETS)
    channel = torch.randn(len(BLOCKS), TINY_HIDDEN, generator=torch.Generator().manual_seed(4))
    states = _states_on_the_channel(channel)
    yes, no = NOUL.option_keys().index("yes"), null_option_index(NOUL.option_keys())
    nulls = torch.full((STATES,), no)
    many = compiled.take(torch.zeros(STATES, dtype=torch.long))
    logits = decision_logits(many, states, channel, nulls)
    facet_logits = torch.einsum("nkbfh,nbh->nbkf", many.detectors, states) + many.offsets.permute(0, 2, 1, 3)
    torch.testing.assert_close(logits[:, :, no], facet_logits[:, :, no, 0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(logits[:, :, yes], torch.logsumexp(facet_logits[:, :, yes], -1), atol=1e-4, rtol=1e-4)
    for facet in range(FACETS):
        gate = firing_gate(compiled, 0, yes, no, BLOCKS[0], channel[0], TEMPERATURE, GATE, facet=facet)
        expected = (facet_logits[:, 0, yes, facet] - facet_logits[:, 0, no, 0]) / TEMPERATURE - logit(GATE)
        torch.testing.assert_close(states[:, 0] @ gate, expected, atol=1e-3, rtol=1e-4)


def test_a_circuit_reads_two_blocks_and_answers_once(reader: RuleReader) -> None:
    from rein.compiler.circuit import CircuitCompiler

    torch.manual_seed(2)
    circuit = CircuitCompiler(TINY_HIDDEN, BLOCKS, units=4, dim=DIM, heads=HEADS, layers=LAYERS, rank=RANK)
    reading = reader.read([(CHOICE, BASE), (NOUL, BASE)])
    states = torch.randn(5, len(BLOCKS), TINY_HIDDEN)
    logits = circuit.item_logits(reading, torch.tensor([0, 1, 1, 0, 1]), states, torch.randn(len(BLOCKS), TINY_HIDDEN))
    assert logits.shape == (5, 1, reading.mask.shape[1]) and circuit.output_blocks == (BLOCKS[1],)
    logits[logits > torch.finfo(logits.dtype).min].sum().backward()
    assert circuit.first_basis.grad is not None and circuit.first_basis.grad.abs().sum() > 0
