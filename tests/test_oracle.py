import torch

from rein.compiler.oracle_probe import choose_l2, fit_probe, probe_probabilities

POINTS, HIDDEN = 120, 16


def test_a_per_rule_probe_separates_a_linear_rule_it_was_trained_on() -> None:
    generator = torch.Generator().manual_seed(0)
    direction = torch.randn(HIDDEN, generator=generator)
    features = torch.randn(POINTS, HIDDEN, generator=generator)
    labels = (features @ direction > 0).long()
    l2 = choose_l2(features[:80], labels[:80], classes=2)
    weights = fit_probe(features[:80], labels[:80], classes=2, l2=l2)
    predicted = probe_probabilities(weights, features[80:]).argmax(-1)
    assert float((predicted == labels[80:]).float().mean()) >= 0.85
