import torch

FORCING_STEPS = 150
FORCING_LEARNING_RATE = 0.05


def forcing_margin(
    embedding: torch.Tensor,
    final_norm_weight: torch.Tensor,
    token: int,
    steps: int = FORCING_STEPS,
    learning_rate: float = FORCING_LEARNING_RATE,
) -> float:
    """The best logit margin of `token` over every other token that one residual direction reaches, per unit of the
    normalized hidden state.

    With tied embeddings, a direction can push a token to the top only if its output row is a vertex of the rows'
    convex hull. A token whose margin stays negative can never be forced by a neuron, so it cannot be a signal.
    """
    rows = embedding.float() * (1.0 + final_norm_weight.float())
    direction = rows[token].clone().requires_grad_(True)
    optimizer = torch.optim.Adam([direction], lr=learning_rate)
    others = torch.ones(rows.shape[0], dtype=torch.bool)
    others[token] = False
    for _ in range(steps):
        logits = rows @ (direction / direction.norm())
        loss = torch.logsumexp(logits[others], dim=0) - logits[token]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = rows @ (direction / direction.norm())
        return float(logits[token] - logits[others].max())
