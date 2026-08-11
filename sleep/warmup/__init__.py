"""
Retrieval-aware warm-up phase (the paper's diagnostic-to-solution extension).

The SLEEP paper diagnoses the recognition--recall gap but does not close it.
This module implements the proposed fix: a one-time warm-up phase, run *before*
any SLEEP continual learning, that teaches a frozen pretrained model the skill
of *using* KV memory during autoregressive generation — without teaching any
specific fact.

Mechanism (Memorizing-Transformers-style, with a G-MemLLM-style trainable
gate):

  1. Take a general text corpus (NOT the evaluation facts).
  2. For each training step, split a corpus sequence into a *prefix* and a
     *continuation*. Store the prefix's episode K/V in the memory bank and
     train the model to predict the continuation with that memory injected.
     Because the prefix is genuinely relevant context for the continuation
     (same document) but is present only through memory, the model must learn
     to route memory into generation to lower its loss.
  3. The only trainable parameters are a small :class:`MemoryGate` on the
     memory value contribution (optionally also the ``w_cons`` LoRA adapter).
     The gate is then frozen and, crucially, lives on the injector rather than
     in ``w_cons`` — so it *persists* through subsequent sleep cycles, which
     only overwrite ``w_cons``.

At evaluation time the fact episodes occupy the bank and the trained gate
amplifies their contribution to generation, which is the prediction the
extension tests: the recognition--recall gap narrows.

Public API:
    MemoryGate     — the trainable gate module (see :mod:`sleep.warmup.gate`).
    WarmupTrainer  — orchestrates the warm-up loop.
    WarmupResult   — dataclass summarising a warm-up run.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sleep.utils.logging import get_logger, metrics
from sleep.warmup.gate import MemoryGate

if TYPE_CHECKING:
    from sleep.weights import DualWeightSystem

logger = get_logger("sleep.warmup")

__all__ = ["MemoryGate", "WarmupTrainer", "WarmupResult"]


@dataclass
class WarmupResult:
    """Summary of a warm-up run.

    Attributes:
        n_steps:      Number of optimizer steps actually taken.
        final_loss:   Loss on the last step.
        mean_loss:    Mean loss across all steps.
        initial_loss: Loss on the first step (for a quick did-it-learn check;
                      ``mean``/``final`` below ``initial`` indicates the gate is
                      learning to use memory).
        gate_scales:  Final per-layer mean multiplicative scale
                      ``exp(log_scale).mean(dim=head)``, keyed by layer index.
                      Values > 1 mean the warm-up learned to amplify memory.
        train_wcons:  Whether ``w_cons`` was trained alongside the gate.
    """

    n_steps: int
    final_loss: float
    mean_loss: float
    initial_loss: float
    gate_scales: dict[int, float] = field(default_factory=dict)
    train_wcons: bool = False

    def as_dict(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "final_loss": self.final_loss,
            "mean_loss": self.mean_loss,
            "initial_loss": self.initial_loss,
            "gate_scales": self.gate_scales,
            "train_wcons": self.train_wcons,
        }


class WarmupTrainer:
    """Trains a :class:`MemoryGate` to use KV memory during generation.

    Args:
        dual_weights: A :class:`sleep.weights.DualWeightSystem` constructed with
            ``use_kv_memory_for_fast=True`` (the KV bank + injector must exist).
        tokenizer:    HuggingFace tokenizer (only its ``pad_token_id`` is used).
        device:       Device string.
        gate_init_scale: Initial gate scale; ``1.0`` (identity) by default so an
            untrained gate is a no-op.
        train_wcons:  If ``True``, also train the ``w_cons`` LoRA adapter during
            warm-up (a stronger but heavier variant). Default ``False`` trains
            only the ~hundreds-of-parameters gate. Note that if ``w_cons`` is
            trained here, a subsequent sleep cycle will continue updating it;
            the gate is what reliably persists.

    Raises:
        RuntimeError: If ``dual_weights`` has no KV memory bank/injector.
    """

    def __init__(
        self,
        dual_weights: "DualWeightSystem",
        tokenizer,
        device: str = "cpu",
        *,
        gate_init_scale: float = 1.0,
        train_wcons: bool = False,
    ) -> None:
        if not getattr(dual_weights, "use_kv_memory_for_fast", False):
            raise RuntimeError(
                "WarmupTrainer requires a DualWeightSystem built with "
                "use_kv_memory_for_fast=True (it needs the KV bank + injector)."
            )
        if dual_weights.kv_injector is None or dual_weights.kv_bank is None:
            raise RuntimeError("KV injector/bank are not initialised.")

        self._dw = dual_weights
        self._tokenizer = tokenizer
        self._device = device
        self._train_wcons = bool(train_wcons)

        num_kv_heads = dual_weights.kv_bank.num_kv_heads
        # The gate rides the injection path, so it is keyed to the injection
        # layer set (which equals adapted_layers under default config but
        # diverges when the consolidation adapter moves elsewhere).
        gate_layers = getattr(dual_weights, "injection_layers", None)
        if gate_layers is None:
            gate_layers = dual_weights.adapted_layers
        self._gate = MemoryGate(
            adapted_layers=gate_layers,
            num_kv_heads=num_kv_heads,
            init_scale=gate_init_scale,
        ).to(device)

    @property
    def gate(self) -> MemoryGate:
        """The gate module (installed on the injector during :meth:`run`)."""
        return self._gate

    def _split_point(self, seq_len: int, rng: random.Random) -> int:
        """Choose a prefix/continuation split so both parts are non-trivial.

        Returns the number of prefix tokens, in ``[1, seq_len-1]``, biased to
        the middle third so neither the memory (prefix) nor the training target
        (continuation) is degenerately short.
        """
        lo = max(1, seq_len // 4)
        hi = max(lo + 1, (3 * seq_len) // 4)
        hi = min(hi, seq_len - 1)
        if hi <= lo:
            return max(1, seq_len // 2)
        return rng.randint(lo, hi)

    def run(
        self,
        corpus: list[torch.Tensor],
        n_steps: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip_norm: float = 1.0,
        min_seq_len: int = 8,
        seed: int = 0,
    ) -> WarmupResult:
        """Run the warm-up loop and leave the trained gate installed + frozen.

        Args:
            corpus:        List of 1-D ``LongTensor`` token sequences from
                           general text (must exclude the evaluation facts).
            n_steps:       Optimizer steps.
            lr:            Learning rate (AdamW). The gate is tiny so a larger LR
                           than sleep training is appropriate.
            weight_decay:  AdamW weight decay (0 by default — we do not want to
                           pull the gate back toward identity).
            grad_clip_norm: Max gradient norm.
            min_seq_len:   Skip corpus sequences shorter than this.
            seed:          Seed for the local RNG that picks sequences/splits.

        Returns:
            A :class:`WarmupResult`. On return, the gate is installed on the
            injector, frozen (``requires_grad=False``), and the warm-up entries
            have been cleared from the bank so evaluation starts from an empty
            bank.

        Raises:
            ValueError: If no corpus sequence meets ``min_seq_len``.
        """
        usable = [s for s in corpus if s.numel() >= min_seq_len]
        if not usable:
            raise ValueError(
                f"No corpus sequence has >= {min_seq_len} tokens; "
                "cannot run warm-up."
            )

        rng = random.Random(seed)
        injector = self._dw.kv_injector
        model = self._dw.model

        # Install the gate and make sure injection is on during warm-up
        # (the inverse of sleep training, which disables it).
        injector.set_memory_gate(self._gate)
        self._dw.set_kv_enabled(True)

        # Build the trainable parameter set.
        for p in self._gate.parameters():
            p.requires_grad_(True)
        trainable = list(self._gate.parameters())
        if self._train_wcons:
            self._dw.set_mode("sleep_training")  # makes w_cons trainable
            trainable += self._dw.get_cons_trainable_params()
        else:
            # Gate-only: base + adapters stay frozen; eval() for determinism.
            model.eval()

        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        pad_id = getattr(self._tokenizer, "pad_token_id", None) or 0

        warmup_tag = "_warmup_episode"
        losses: list[float] = []

        for step in range(1, n_steps + 1):
            seq = rng.choice(usable).to(self._device)
            seq_len = int(seq.numel())
            split = self._split_point(seq_len, rng)

            # Store the prefix [0, split) as episode memory. Evict the previous
            # step's entry first so the bank holds only the current prefix.
            self._dw.evict_from_kv_bank(warmup_tag)
            try:
                self._dw.write_to_kv_bank(
                    tag_id=warmup_tag,
                    token_ids=seq,
                    span_start=0,
                    span_end=split,
                    device=self._device,
                )
            except Exception as exc:  # bank capacity or extraction hiccup
                logger.debug("warm-up step %d: skipped (write failed: %s)", step, exc)
                continue

            # Train on the continuation [split:] with the prefix in memory.
            continuation = seq[split:].unsqueeze(0)  # (1, cont_len)
            if continuation.shape[1] < 2:
                continue
            labels = continuation.clone()
            labels[labels == pad_id] = -100

            outputs = model(input_ids=continuation, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
            optimizer.step()

            loss_val = float(loss.item())
            losses.append(loss_val)
            if step == 1 or step % max(1, n_steps // 10) == 0:
                logger.info(
                    "warm-up step %d/%d | loss=%.4f | mean_gate_scale=%.3f",
                    step, n_steps, loss_val, self._mean_gate_scale(),
                )
                metrics.log({"warmup/loss": loss_val, "warmup/step": step})

        # Freeze the gate and clear the warm-up memory so evaluation starts
        # from an empty bank; the gate persists on the injector.
        for p in self._gate.parameters():
            p.requires_grad_(False)
        self._dw.evict_from_kv_bank(warmup_tag)

        result = WarmupResult(
            n_steps=len(losses),
            final_loss=losses[-1] if losses else 0.0,
            mean_loss=(sum(losses) / len(losses)) if losses else 0.0,
            initial_loss=losses[0] if losses else 0.0,
            gate_scales=self._gate_scales(),
            train_wcons=self._train_wcons,
        )
        logger.info(
            "Warm-up complete | steps=%d | loss %.4f -> %.4f | mean gate scale=%.3f",
            result.n_steps, result.initial_loss, result.final_loss,
            self._mean_gate_scale(),
        )
        return result

    def _gate_scales(self) -> dict[int, float]:
        """Per-layer mean multiplicative scale (for logging/diagnostics)."""
        out: dict[int, float] = {}
        with torch.no_grad():
            for layer_idx in self._gate.adapted_layers:
                out[layer_idx] = float(self._gate.scale_for_layer(layer_idx).mean().item())
        return out

    def _mean_gate_scale(self) -> float:
        scales = self._gate_scales()
        return sum(scales.values()) / max(len(scales), 1)
