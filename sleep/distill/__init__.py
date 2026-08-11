"""
Context distillation for consolidation (the localisation revision's decoupler).

The paper's two failures are independent: KV memory cannot steer generation
(Section 7.2), and consolidation cannot write facts into weights (7.4). The
original pipeline chained them — consolidation learned from the KV pathway, so
the first failure contaminated the test of the second.

This module removes the KV bank from consolidation entirely. The teacher is
the model's own in-context behaviour, which our baselines show is strong
(DRA 0.760 on Qwen-7B, 0.987 on Llama-8B): place the fact in the prompt, take
the full next-token distribution over an answer text, and distil that
distribution into an adapter evaluated *without* the fact in the prompt.

    teacher:  [Context: <fact>\\n] <target text>   (adapters disabled, no grad)
    student:                       <target text>   (adapter active, trained)

    loss = KL(teacher_T || student_T) * T^2  +  alpha_ce * CE(student, gold)

aligned on the target tokens the two sides share. Combined with paraphrase
diversity (each fact distilled through many wordings and QA forms) and a
mid-stack MLP adapter placement, this is the direct test of the localisation
hypothesis: if facts fail to become extractable even when a proven teacher
writes them through the mechanism the editing literature says is correct, the
failure is deeper than placement.

Public API:
    ContextDistiller — orchestrates the distillation loop.
    DistillResult    — summary statistics.
"""

from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sleep.utils.logging import get_logger, metrics

logger = get_logger("sleep.distill")

__all__ = ["ContextDistiller", "DistillResult"]


@dataclass
class DistillResult:
    """Summary of one distillation run.

    Attributes:
        n_steps:      Optimizer steps taken.
        initial_loss: Total loss at the first step.
        mean_loss:    Mean total loss over all steps.
        final_loss:   Total loss at the last step.
        mean_kl:      Mean KL component.
        mean_ce:      Mean CE component.
    """

    n_steps: int
    initial_loss: float
    mean_loss: float
    final_loss: float
    mean_kl: float
    mean_ce: float

    def as_dict(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "initial_loss": self.initial_loss,
            "mean_loss": self.mean_loss,
            "final_loss": self.final_loss,
            "mean_kl": self.mean_kl,
            "mean_ce": self.mean_ce,
        }


class ContextDistiller:
    """Distil in-context knowledge into an adapter, no memory injection.

    Args:
        model:     A causal LM whose trainable parameters are exactly the
            adapter to be written (e.g. a ``PeftModel`` in a mode where only
            the target adapter has ``requires_grad=True``). If the model
            exposes ``disable_adapter()`` (PEFT), the teacher pass runs inside
            it so the teacher is the clean base model; otherwise the teacher
            is the same model (caller must ensure that is intended).
        tokenizer: Matching tokenizer.
        device:    Device string.
        kd_temperature: Softmax temperature ``T`` for the KL term. The loss is
            scaled by ``T**2`` per standard practice.
        alpha_ce:  Weight of the hard-label CE term mixed with the KL.
        context_template: How the fact is presented to the teacher. ``{fact}``
            is replaced with the fact text.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        *,
        kd_temperature: float = 2.0,
        alpha_ce: float = 0.5,
        context_template: str = "Context: {fact}\n",
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._T = float(kd_temperature)
        self._alpha_ce = float(alpha_ce)
        self._context_template = context_template

    # -- internals -----------------------------------------------------------

    def _teacher_context(self):
        """Context manager under which the teacher forward runs."""
        disable = getattr(self._model, "disable_adapter", None)
        return disable() if callable(disable) else nullcontext()

    def _encode(self, text: str) -> torch.Tensor:
        return self._tokenizer(text, return_tensors="pt").input_ids.to(self._device)

    def step_loss(self, fact_text: str, target: str) -> tuple[torch.Tensor, float, float]:
        """Compute the distillation loss for one (fact, target-wording) pair.

        Returns ``(total_loss, kl_value, ce_value)``.
        """
        prefix = self._context_template.format(fact=fact_text)
        prefix_ids = self._encode(prefix)
        target_ids = self._encode(target)
        n_t = target_ids.shape[1]
        if n_t < 2:
            raise ValueError("Target must be at least 2 tokens for next-token loss.")

        # Teacher: base model (adapters disabled), fact in context, no grad.
        teacher_input = torch.cat([prefix_ids, target_ids], dim=1)
        with torch.no_grad(), self._teacher_context():
            t_logits = self._model(input_ids=teacher_input).logits
        off = prefix_ids.shape[1]
        # Predictions for target tokens t_1..t_{n-1}: teacher logits at
        # positions [off, off+n_t-2] predict exactly those tokens.
        t_pred = t_logits[:, off - 1 + 1 : off + n_t - 1, :]

        # Student: adapter active, NO context.
        s_logits = self._model(input_ids=target_ids).logits
        # Student logits at [0, n_t-2] predict t_1..t_{n-1}.
        s_pred = s_logits[:, : n_t - 1, :]

        gold = target_ids[:, 1:]

        T = self._T
        vocab = s_pred.shape[-1]
        # Flatten to (n_positions, vocab) so "batchmean" divides by the number
        # of token positions — i.e. a per-token KL, on the same scale as the
        # CE term. Without the flatten, batchmean divides by the batch size
        # (1) and the KL sums over the whole sequence, dwarfing the CE and
        # mis-scaling gradients.
        kl = F.kl_div(
            F.log_softmax(s_pred.float().reshape(-1, vocab) / T, dim=-1),
            F.log_softmax(t_pred.float().reshape(-1, vocab) / T, dim=-1),
            log_target=True,
            reduction="batchmean",
        ) * (T * T)
        ce = F.cross_entropy(
            s_pred.float().reshape(-1, s_pred.shape[-1]), gold.reshape(-1),
        )
        total = kl + self._alpha_ce * ce
        return total, float(kl.item()), float(ce.item())

    # Backward-compatible private alias (tests spy on this seam).
    @property
    def _step_loss(self):
        return self.step_loss

    @_step_loss.setter
    def _step_loss(self, fn):
        self.step_loss = fn

    # -- main loop -----------------------------------------------------------

    def run(
        self,
        facts: list[dict],
        n_steps: int = 400,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        grad_clip_norm: float = 1.0,
        seed: int = 0,
        use_paraphrases: bool = True,
        log_every: int | None = None,
    ) -> DistillResult:
        """Run the distillation loop over a fact set.

        Each step samples one fact and one of its wordings (the original text,
        or — when ``use_paraphrases`` and the fact carries a ``paraphrases``
        list — a uniformly drawn paraphrase/QA form), and takes one optimizer
        step on the combined KD loss. Sampling is seeded and independent of
        global RNG state.

        Args:
            facts: Fact dicts with ``text`` (and optionally ``paraphrases``).
            n_steps: Total optimizer steps.
            lr, weight_decay, grad_clip_norm: AdamW settings.
            seed: Seed for the local sampling RNG.
            use_paraphrases: If ``False``, always train on ``fact["text"]``
                (the single-wording ablation arm).
            log_every: Log cadence; defaults to ``n_steps // 10``.

        Returns:
            A :class:`DistillResult`.

        Raises:
            ValueError: If ``facts`` is empty or no parameter is trainable.
        """
        if not facts:
            raise ValueError("facts must be non-empty")

        trainable = [p for p in self._model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError(
                "No trainable parameters. Put the model in a mode where the "
                "target adapter has requires_grad=True before distilling."
            )
        n_trainable = sum(p.numel() for p in trainable)
        logger.info(
            "ContextDistiller: %d facts | %d steps | lr=%g | T=%g | alpha_ce=%g "
            "| paraphrases=%s | trainable=%s",
            len(facts), n_steps, lr, self._T, self._alpha_ce,
            use_paraphrases, f"{n_trainable:,}",
        )

        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        rng = random.Random(seed)
        if log_every is None:
            log_every = max(1, n_steps // 10)

        losses: list[float] = []
        kls: list[float] = []
        ces: list[float] = []

        for step in range(1, n_steps + 1):
            fact = rng.choice(facts)
            wordings = [fact["text"]]
            if use_paraphrases and fact.get("paraphrases"):
                wordings = fact["paraphrases"]
            target = rng.choice(wordings)

            total, kl_v, ce_v = self.step_loss(fact["text"], target)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
            optimizer.step()

            losses.append(float(total.item()))
            kls.append(kl_v)
            ces.append(ce_v)

            if step == 1 or step % log_every == 0:
                logger.info(
                    "distill step %d/%d | loss=%.4f (kl=%.4f ce=%.4f)",
                    step, n_steps, losses[-1], kl_v, ce_v,
                )
                metrics.log({"distill/loss": losses[-1], "distill/step": step})

        result = DistillResult(
            n_steps=len(losses),
            initial_loss=losses[0],
            mean_loss=sum(losses) / len(losses),
            final_loss=losses[-1],
            mean_kl=sum(kls) / len(kls),
            mean_ce=sum(ces) / len(ces),
        )
        logger.info(
            "Distillation complete | steps=%d | loss %.4f -> %.4f (mean %.4f)",
            result.n_steps, result.initial_loss, result.final_loss, result.mean_loss,
        )
        return result
