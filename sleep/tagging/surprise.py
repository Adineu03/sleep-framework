"""
Per-token surprise computation (Q1.2, Step 1).

Computes Shannon surprise e_t = -log p_{W_slow}(x_t | x_{<t}) for each token
in a sequence, using a single forward pass through a HuggingFace causal LM.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


@dataclass
class SurpriseResult:
    """Output of per-token surprise computation.

    Attributes:
        surprises: Per-token surprise in nats.  Length = seq_len.
            The first token has surprise 0 (no preceding context).
        hidden_states: Per-token final-layer hidden states, each of shape (d_model,).
            Length = seq_len.
        tokens: The input token IDs (1-D tensor of shape (seq_len,)).
    """

    surprises: list[float]
    hidden_states: list[torch.Tensor]
    tokens: torch.Tensor


def compute_surprise(
    model: PreTrainedModel,
    token_ids: torch.Tensor,  # shape (seq_len,)
    device: str = "cpu",
) -> SurpriseResult:
    """Run a single forward pass and return per-token surprise + hidden states.

    Parameters
    ----------
    model:
        A HuggingFace causal language model (e.g. GPT-2, LLaMA).
        Must support ``output_hidden_states=True``.
    token_ids:
        1-D tensor of token IDs, shape ``(seq_len,)``.
    device:
        Device string forwarded to tensors (``"cpu"``, ``"cuda"``, etc.).

    Returns
    -------
    SurpriseResult
        Contains per-token surprises (nats), final-layer hidden states,
        and the original token IDs.
    """
    token_ids = token_ids.to(device)
    model = model.to(device)
    model.eval()

    seq_len = token_ids.shape[0]

    # Single forward pass — no gradients needed (inference only).
    with torch.no_grad():
        outputs = model(
            input_ids=token_ids.unsqueeze(0),  # (1, seq_len)
            output_hidden_states=True,
        )

    # logits: (1, seq_len, vocab_size)
    logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Final-layer hidden states: (1, seq_len, d_model)
    final_hidden = outputs.hidden_states[-1][0]  # (seq_len, d_model)

    # Compute log-softmax over vocab dimension.
    log_probs = F.log_softmax(logits, dim=-1)  # (seq_len, vocab_size)

    # Per-token surprise: e_t = -log p(x_t | x_{<t})
    # logits[t-1] predicts token_ids[t], so surprise for token t uses
    # log_probs[t-1] indexed by token_ids[t].
    surprises: list[float] = [0.0]  # first token has no prediction context
    for t in range(1, seq_len):
        neg_ll = -log_probs[t - 1, token_ids[t]].item()
        surprises.append(neg_ll)

    # Collect per-token hidden states as individual (d_model,) tensors.
    hidden_states: list[torch.Tensor] = [
        final_hidden[t].clone() for t in range(seq_len)
    ]

    return SurpriseResult(
        surprises=surprises,
        hidden_states=hidden_states,
        tokens=token_ids.cpu(),
    )