"""Next-code cross-entropy loss for backbone pretraining."""

import torch
import torch.nn.functional as F


def next_token_loss(
    logits: torch.Tensor,
    target_codes: torch.Tensor,
) -> torch.Tensor:
    prediction_logits = logits[:, :-1].float()
    prediction_targets = target_codes[:, 1:].long()
    return F.cross_entropy(
        prediction_logits.flatten(0, 1),
        prediction_targets.flatten(),
        ignore_index=-100,
    )
