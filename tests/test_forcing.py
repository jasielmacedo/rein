import torch

from rein.neurons.forcing import forcing_margin

HIDDEN = 6
VERTICES = 8


def test_a_token_inside_the_hull_of_the_others_can_never_be_forced() -> None:
    generator = torch.Generator().manual_seed(0)
    vertices = torch.nn.functional.normalize(torch.randn(VERTICES, HIDDEN, generator=generator), dim=-1)
    interior = 0.3 * (vertices[0] + vertices[1] + vertices[2]) / 3
    embedding = torch.cat([vertices, interior[None]])
    norm_weight = torch.zeros(HIDDEN)
    assert forcing_margin(embedding, norm_weight, token=VERTICES) < 0
    assert forcing_margin(embedding, norm_weight, token=0) > 0
