"""
Retrieval-aware memory gate (the warm-up extension's trainable bridge).

The SLEEP paper's central negative finding is the recognition--recall gap: a
frozen pretrained model can let injected key--value (KV) memory bias a
four-way classification (Tagged--Untagged $\\Delta = +0.16$ MC) but cannot let
it steer free-form autoregressive generation. The mechanistic cause is that
memory enters as a small attention-time perturbation against a dominant
pretrained prior.

This module is the minimal, principled intervention proposed to narrow that
gap, grounded in gated-memory work (G-MemLLM, MemoryLLM): a small trainable
gate on the *memory value contribution*. Because attention output is linear in
V and the softmax weights depend only on Q$\\cdot$K (not on V), a per-kv-head
scale on the memory values exactly modulates how strongly retrieved memory
feeds the residual stream:

    attn = \\sum_{j\\in cur} w_j V_j + \\sum_{j\\in mem} w_j (g \\odot V_j)
         = current_contribution + (g \\odot memory_contribution).

The gate holds one scale per (adapted layer, kv-head), parameterised as a
log-scale initialised to zero so that ``exp(0) = 1`` makes an untrained gate an
exact identity — installing it changes nothing until the warm-up phase trains
it. Total parameters are ``n_adapted_layers * num_kv_heads`` (e.g. 9 * 4 = 36
for the top-third of Qwen2.5-7B): the "few hundred parameters" bridge the
extension proposal describes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sleep.utils.logging import get_logger

logger = get_logger("sleep.warmup.gate")

__all__ = ["MemoryGate"]


class MemoryGate(nn.Module):
    """Per-layer, per-kv-head multiplicative gate on injected memory values.

    Installed on a :class:`sleep.weights.kv_injection.KVInjector` via
    ``injector.set_memory_gate(gate)``. The injector calls it as
    ``gate(mem_v_b, layer_idx=...)`` with ``mem_v_b`` of shape
    ``(batch, num_kv_heads, n_mem, head_dim)`` and expects the same shape back.

    Args:
        adapted_layers: The layer indices the injector serves (``bank
            .adapted_layers``). The gate keeps one parameter row per layer, in
            this order, so ``layer_idx`` values passed at call time must be a
            subset of these.
        num_kv_heads:   Number of key/value heads (GQA). One scale per head.
        init_scale:     Initial multiplicative scale (default ``1.0`` =
            identity). Values != 1.0 are supported for ablations but the warm-up
            should start from identity so an untrained gate is a no-op.

    Notes:
        A scale of exactly ``1.0`` at every entry makes the gated forward
        bit-identical to the ungated one — the equality relied on by the
        injector's gate test. Scaling memory values does **not** reclaim the
        softmax mass that memory positions absorb, so ``g -> 0`` does not
        reproduce the exact no-injection output; this gate is designed to learn
        to *amplify* useful memory (``g > 1``) during generation, which is the
        direction that narrows the recognition--recall gap.
    """

    def __init__(
        self,
        adapted_layers: list[int],
        num_kv_heads: int,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if not adapted_layers:
            raise ValueError("adapted_layers must be non-empty")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
        if init_scale <= 0:
            raise ValueError(f"init_scale must be positive, got {init_scale}")

        # Preserve the injector's layer order; map layer_idx -> row.
        self._layers: list[int] = list(adapted_layers)
        self._layer_to_row: dict[int, int] = {
            layer_idx: row for row, layer_idx in enumerate(self._layers)
        }
        self.num_kv_heads = int(num_kv_heads)

        # Parameterise as a log-scale so the scale stays positive and identity
        # is exactly log_scale == 0. Shape: (n_layers, num_kv_heads).
        import math

        init_log = math.log(init_scale)
        self.log_scale = nn.Parameter(
            torch.full((len(self._layers), self.num_kv_heads), init_log)
        )

    @property
    def adapted_layers(self) -> list[int]:
        """Layer indices this gate holds parameters for, in row order."""
        return list(self._layers)

    def scale_for_layer(self, layer_idx: int) -> torch.Tensor:
        """Return the positive per-head scale vector for one layer.

        Args:
            layer_idx: A layer index present in ``adapted_layers``.

        Returns:
            1-D tensor of shape ``(num_kv_heads,)`` = ``exp(log_scale[row])``.

        Raises:
            KeyError: If ``layer_idx`` was not in ``adapted_layers``.
        """
        row = self._layer_to_row[layer_idx]
        return torch.exp(self.log_scale[row])

    def forward(self, mem_v: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Scale memory values for one layer's attention.

        Args:
            mem_v:     Memory value tensor of shape
                       ``(batch, num_kv_heads, n_mem, head_dim)``.
            layer_idx: The decoder layer this call is for.

        Returns:
            ``mem_v`` scaled per kv-head, same shape and dtype.
        """
        scale = self.scale_for_layer(layer_idx)  # (num_kv_heads,)
        # Broadcast over (batch, num_kv_heads, n_mem, head_dim).
        scale = scale.to(dtype=mem_v.dtype, device=mem_v.device)
        return mem_v * scale.view(1, -1, 1, 1)
