"""
SLEEP Engine — orchestrates one complete sleep consolidation cycle.

Phases:
    1. GENERATE  — create replay samples from PRP-allocated tags
    2. QUALITY   — filter replays through semantic similarity + surprise checks
    3. TRAIN     — run sleep training loop on W_cons
    4. PPL CHECK — optional rollback if benchmark perplexity degrades
    5. VALIDATE & CLEANUP — confirm consolidation, remove/penalise tags

Public API:
    SleepEngine          — main orchestrator class
    ReplaySample         — dataclass for a single replay sample
    generate_replay      — generate one replay from a tag
    quality_check        — check replay quality
    sleep_train          — run the sleep training loop
    compute_fisher_diagonal — estimate diagonal Fisher information
    compute_ewc_loss     — compute EWC regularisation penalty
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sleep.config import SleepConfig, WeightsConfig
from sleep.sleep_engine.cleanup import (
    cleanup_tags,
    compute_span_surprise,
    validate_consolidation,
)
from sleep.sleep_engine.fisher import compute_ewc_loss, compute_fisher_diagonal
from sleep.sleep_engine.quality import compute_baseline_surprise, quality_check
from sleep.sleep_engine.replay import ReplaySample, generate_replay
from sleep.sleep_engine.train import sleep_train
from sleep.utils.logging import get_logger, metrics

if TYPE_CHECKING:
    from sleep.tagging.tags import Tag, TagKeyProjection
    from sleep.weights import DualWeightSystem

logger = get_logger("sleep.sleep_engine")


# ---------------------------------------------------------------------------
# PPL evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_ppl(
    model: torch.nn.Module,
    benchmark_data: list[torch.Tensor],
    device: str = "cpu",
) -> float:
    """Compute perplexity of *model* on benchmark sequences.

    Args:
        model:          A causal LM in eval mode.
        benchmark_data: List of 1-D token-ID tensors.
        device:         Device for forward passes.

    Returns:
        Perplexity (float).  Returns ``float('inf')`` when no valid tokens
        are available.
    """
    total_nll: float = 0.0
    total_tokens: int = 0

    for seq in benchmark_data:
        if seq.numel() < 2:
            continue

        input_ids = seq.unsqueeze(0).to(device)  # (1, seq_len)
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_nll += loss.item()
        total_tokens += shift_labels.numel()

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    return float(torch.exp(torch.tensor(avg_nll)).item())


# ---------------------------------------------------------------------------
# SleepEngine
# ---------------------------------------------------------------------------

class SleepEngine:
    """Orchestrates a complete sleep consolidation cycle.

    The engine coordinates replay generation, quality filtering, training,
    validation, and cleanup into a single :meth:`run_cycle` call.

    Args:
        dual_weights:   The :class:`DualWeightSystem` managing adapters.
        tokenizer:      HuggingFace tokenizer matching the model.
        sleep_config:   :class:`SleepConfig` with sleep-phase hyperparameters.
        weights_config: :class:`WeightsConfig` with LR, EWC, PPL-safety params.
        mu_surprise:    Baseline per-token surprise (nats).  If ``None``,
                        must be set later via :meth:`set_baseline`.
        device:         Device string (``"cpu"``, ``"cuda:0"``, etc.).
    """

    def __init__(
        self,
        dual_weights: DualWeightSystem,
        tokenizer,
        sleep_config: SleepConfig,
        weights_config: WeightsConfig,
        mu_surprise: float | None = None,
        device: str = "cpu",
        replay_strategy: str = "generative",
    ) -> None:
        """
        Args:
            ...
            replay_strategy: How to construct replays in Phase 1.
                ``"generative"`` (default): autoregressive generation conditioned
                    on KV memory. The original SLEEP design.
                ``"original"``: bypass generation; use the original tagged
                    experience's tokens directly as the replay. Diagnostic
                    mode that strips out replay-generation as a confound;
                    answers "given perfect replays, does W_cons consolidation
                    transfer to free-form recall?"
        """
        valid_strategies = {"generative", "original"}
        if replay_strategy not in valid_strategies:
            raise ValueError(
                f"replay_strategy must be one of {valid_strategies}, "
                f"got {replay_strategy!r}"
            )
        self._dual_weights = dual_weights
        self._tokenizer = tokenizer
        self._sleep_config = sleep_config
        self._weights_config = weights_config
        self._mu_surprise = mu_surprise
        self._device = device
        self._replay_strategy = replay_strategy

    # -- Properties ----------------------------------------------------------

    @property
    def mu_surprise(self) -> float | None:
        """Baseline per-token surprise threshold (nats), or ``None``."""
        return self._mu_surprise

    # -- Baseline calibration ------------------------------------------------

    def set_baseline(self, calibration_data: list[torch.Tensor]) -> float:
        """Compute and set ``mu_surprise`` from calibration data.

        Puts the model into ``target_inference`` mode (W_slow + W_cons) and
        measures mean per-token surprisal on the calibration sequences.

        Args:
            calibration_data: List of 1-D token-ID tensors from general text.

        Returns:
            The computed ``mu_surprise`` value (also stored internally).
        """
        self._dual_weights.set_mode("target_inference")
        model = self._dual_weights.model
        model.eval()

        mu: float = compute_baseline_surprise(
            model, calibration_data, device=self._device,
        )
        self._mu_surprise = mu

        logger.info("Baseline surprise set: mu_surprise=%.4f nats", mu)
        metrics.log({"sleep_engine/mu_surprise": mu})
        return mu

    # -- KV memory control ---------------------------------------------------

    def _set_kv_enabled(self, enabled: bool) -> None:
        """Enable or disable KV memory injection if the dual-weight system
        is using KV memory as the W_fast substrate.

        No-op when KV memory mode is disabled (LoRA W_fast path).
        """
        if getattr(self._dual_weights, "use_kv_memory_for_fast", False):
            self._dual_weights.set_kv_enabled(enabled)
            logger.info("KV memory injection: %s", "ENABLED" if enabled else "DISABLED")

    def _clear_kv_bank_if_present(self) -> None:
        """Clear the KV memory bank after a successful consolidation cycle.

        After successful sleep training, knowledge has transferred from the
        bank into W_cons; the bank entries are no longer needed and would
        otherwise persist as a confound for the validation phase or for
        the user's downstream evaluation.

        No-op when KV memory mode is disabled.
        """
        if getattr(self._dual_weights, "use_kv_memory_for_fast", False):
            n_before = self._dual_weights.kv_bank.n_tags
            self._dual_weights.clear_kv_bank()
            logger.info("KV bank cleared (%d entries removed)", n_before)

    # -- Main cycle ----------------------------------------------------------

    def run_cycle(
        self,
        candidates: list[Tag],
        original_tokens_map: dict,
        benchmark_data: list[torch.Tensor] | None = None,
        key_projection: TagKeyProjection | None = None,
    ) -> dict:
        """Execute one complete sleep cycle.

        Phases:
            1. **GENERATE** — create replay samples from each candidate tag.
            2. **QUALITY** — filter replays through quality check.
            3. **TRAIN** — run sleep training loop on W_cons.
            4. **PPL CHECK** — optional; rollback if benchmark PPL degrades.
            5. **VALIDATE & CLEANUP** — confirm consolidation, penalise failures.

        Args:
            candidates:          PRP-allocated tags from
                                 ``PRPSystem.get_consolidation_candidates``.
            original_tokens_map: ``{source_id: token_tensor}`` for replay
                                 generation and validation.
            benchmark_data:      Optional token tensors for PPL evaluation.
            key_projection:      :class:`TagKeyProjection` for quality checks.
                                 If ``None``, quality checks are skipped.

        Returns:
            A dict with cycle statistics::

                {
                    "n_candidates": int,
                    "n_replays_generated": int,
                    "n_replays_accepted": int,
                    "n_consolidated": int,
                    "n_failed": int,
                    "n_permanently_removed": int,
                    "training_stats": dict,
                    "ppl_before": float | None,
                    "ppl_after": float | None,
                    "rolled_back": bool,
                }
        """
        result: dict = {
            "n_candidates": len(candidates),
            "n_replays_generated": 0,
            "n_replays_accepted": 0,
            "n_consolidated": 0,
            "n_failed": 0,
            "n_permanently_removed": 0,
            "training_stats": {},
            "ppl_before": None,
            "ppl_after": None,
            "rolled_back": False,
            # Diagnostic: samples of generated replays alongside their
            # source-fact context, so callers can compare replay quality
            # against the original. Populated up to 3 entries.
            "replay_samples": [],
        }

        # ---- Early exit: nothing to do ----
        if not candidates:
            logger.info("run_cycle: no candidates — returning early")
            return result

        if self._mu_surprise is None:
            raise RuntimeError(
                "mu_surprise has not been set. "
                "Call set_baseline() before run_cycle()."
            )

        model = self._dual_weights.model
        config = self._sleep_config

        logger.info(
            "=== Sleep cycle START | %d candidates ===", len(candidates),
        )

        # ================================================================
        # Phase 1 — GENERATE
        # ================================================================
        logger.info("Phase 1/5: GENERATE (strategy=%s)", self._replay_strategy)
        self._dual_weights.set_mode("sleep_generation")
        model.eval()

        replay_dataset: list[ReplaySample] = []
        replay_tag_map: list[Tag] = []  # parallel: tag for each replay

        if self._replay_strategy == "generative":
            # KV memory ENABLED during replay generation: the model can attend
            # to stored experiences while producing the synthetic training
            # examples.
            self._set_kv_enabled(True)
            for tag in candidates:
                _span_start, _span_end, source_id = tag.ctx
                original_tokens = original_tokens_map.get(source_id)

                sample: ReplaySample | None = None
                for _attempt in range(config.max_generation_attempts):
                    sample = generate_replay(
                        tag=tag,
                        model=model,
                        tokenizer=self._tokenizer,
                        original_tokens=original_tokens,
                        config=config,
                        device=self._device,
                    )
                    if sample is not None:
                        break

                if sample is not None:
                    replay_dataset.append(sample)
                    replay_tag_map.append(tag)
        elif self._replay_strategy == "original":
            # Diagnostic mode: bypass generation entirely. Use the original
            # tagged experience's tokens directly as the "replay". This
            # strips out replay-generation as a confound and asks: given
            # perfect replays, does W_cons consolidation transfer to recall?
            #
            # We deduplicate by source_id — multiple tags pointing at the
            # same experience produce only one replay (the experience).
            self._set_kv_enabled(False)  # KV not needed; we have the source tokens
            seen_sources: set = set()
            for tag in candidates:
                _span_start, _span_end, source_id = tag.ctx
                if source_id in seen_sources:
                    continue
                original_tokens = original_tokens_map.get(source_id)
                if original_tokens is None:
                    logger.debug(
                        "Skipping original-replay for tag %d: no source tokens",
                        id(tag),
                    )
                    continue
                seen_sources.add(source_id)
                # The replay IS the original. seed_length = full length means
                # zero "generated" tokens — every token came from the source.
                full_len = int(original_tokens.shape[0])
                sample = ReplaySample(
                    text_ids=original_tokens.detach().clone(),
                    tag_id=id(tag),
                    prp_score=float(getattr(tag, "p", 1.0)),
                    original_length=full_len,
                    seed_length=full_len,
                )
                replay_dataset.append(sample)
                replay_tag_map.append(tag)
            logger.info(
                "Phase 1 (original strategy): %d unique-source replays from "
                "%d candidates",
                len(replay_dataset), len(candidates),
            )
        else:
            raise RuntimeError(f"unknown replay_strategy {self._replay_strategy!r}")

        result["n_replays_generated"] = len(replay_dataset)

        # Capture up to 3 samples for diagnostic inspection. We decode the
        # tokens here (not in the experiment script) so callers don't need
        # tokenizer-specific knowledge.
        for sample, tag in list(zip(replay_dataset, replay_tag_map))[:3]:
            try:
                _span_start, _span_end, src_id = tag.ctx
                original = original_tokens_map.get(src_id)

                # ReplaySample.text_ids is the full token sequence (seed +
                # generated continuation). seed_length tells us where the
                # seed ends and the model-generated portion begins.
                tokens_flat = sample.text_ids.squeeze().tolist()
                if tokens_flat and isinstance(tokens_flat[0], list):
                    tokens_flat = tokens_flat[0]

                seed_len = int(sample.seed_length)
                seed_text = self._tokenizer.decode(
                    tokens_flat[:seed_len], skip_special_tokens=True,
                )
                generated_text = self._tokenizer.decode(
                    tokens_flat[seed_len:], skip_special_tokens=True,
                )
                full_text = self._tokenizer.decode(
                    tokens_flat, skip_special_tokens=True,
                )
                original_span_text = (
                    self._tokenizer.decode(
                        original[_span_start:_span_end].tolist(),
                        skip_special_tokens=True,
                    ) if original is not None else "<unavailable>"
                )
                result["replay_samples"].append({
                    "source_id": src_id,
                    "original_span": original_span_text,
                    "replay": full_text,
                    "replay_seed": seed_text,
                    "replay_generated": generated_text,
                    "replay_n_tokens": int(sample.text_ids.numel()),
                    "seed_length": seed_len,
                })
            except Exception as exc:
                logger.warning("replay sample decode failed: %s", exc)
                result["replay_samples"].append({
                    "source_id": str(getattr(tag, "ctx", ("?", "?", "?"))[2]),
                    "original_span": "<decode failed>",
                    "replay": f"<decode failed: {exc}>",
                    "replay_n_tokens": -1,
                })

        logger.info(
            "Phase 1 complete: %d/%d replays generated",
            len(replay_dataset), len(candidates),
        )
        metrics.log({
            "sleep_engine/n_replays_generated": len(replay_dataset),
            "sleep_engine/n_candidates": len(candidates),
        })

        # ================================================================
        # Phase 2 — QUALITY CHECK
        # ================================================================
        logger.info("Phase 2/5: QUALITY CHECK")
        self._dual_weights.set_mode("target_inference")
        model.eval()

        accepted: list[ReplaySample] = []
        accepted_tags: list[Tag] = []

        for sample, tag in zip(replay_dataset, replay_tag_map):
            if key_projection is not None:
                passed, reason = quality_check(
                    replay_ids=sample.text_ids,
                    tag=tag,
                    model=model,
                    key_projection=key_projection,
                    config=config,
                    mu_surprise=self._mu_surprise,
                    device=self._device,
                )
            else:
                # Without key_projection we cannot run the quality gate;
                # accept all replays (degraded mode).
                logger.debug(
                    "No key_projection — accepting replay for tag %d "
                    "without quality check",
                    id(tag),
                )
                passed, reason = True, ""

            if passed:
                accepted.append(sample)
                accepted_tags.append(tag)
            else:
                logger.debug(
                    "Rejected replay for tag %d: %s", id(tag), reason,
                )

        result["n_replays_accepted"] = len(accepted)
        logger.info(
            "Phase 2 complete: %d/%d replays accepted",
            len(accepted), len(replay_dataset),
        )
        metrics.log({"sleep_engine/n_replays_accepted": len(accepted)})

        # ---- Early exit: nothing accepted ----
        if not accepted:
            logger.info(
                "No replays accepted — skipping training, "
                "marking all candidates as failed",
            )
            validation_results: dict[int, bool] = {
                id(tag): False for tag in candidates
            }
            cleanup_result = cleanup_tags(
                candidates, validation_results, config,
            )
            result["n_failed"] = len(cleanup_result["failed"])
            result["n_permanently_removed"] = len(
                cleanup_result["permanently_removed"]
            )
            # No consolidation happened: keep bank populated for retry,
            # restore enabled state for next cycle.
            self._set_kv_enabled(True)
            logger.info("=== Sleep cycle END (no accepted replays) ===")
            return result

        # ================================================================
        # Phase 3 — TRAIN
        # ================================================================
        logger.info("Phase 3/5: TRAIN (%d accepted replays)", len(accepted))

        # KV memory DISABLED during training: W_cons must learn to predict
        # replay tokens from context alone, without the memory crutch.
        # If we left it enabled, W_cons would learn to lean on stored K/V
        # that won't exist at inference time (after the bank is cleared).
        self._set_kv_enabled(False)

        # Snapshot W_cons before training (for PPL rollback & validation).
        cons_checkpoint: dict[str, torch.Tensor] = (
            self._dual_weights.save_cons_checkpoint()
        )

        training_stats: dict = sleep_train(
            dual_weights=self._dual_weights,
            replay_dataset=accepted,
            tokenizer=self._tokenizer,
            config=config,
            weights_config=self._weights_config,
            device=self._device,
        )
        result["training_stats"] = training_stats

        logger.info(
            "Phase 3 complete: %d steps, final_loss=%.4f, mean_loss=%.4f",
            training_stats.get("n_steps", 0),
            training_stats.get("final_loss", 0.0),
            training_stats.get("mean_loss", 0.0),
        )
        metrics.log({
            "sleep_engine/train_n_steps": training_stats.get("n_steps", 0),
            "sleep_engine/train_final_loss": training_stats.get("final_loss", 0.0),
            "sleep_engine/train_mean_loss": training_stats.get("mean_loss", 0.0),
        })

        # ================================================================
        # Phase 4 — PPL CHECK (optional)
        # ================================================================
        if benchmark_data is not None and len(benchmark_data) > 0:
            logger.info("Phase 4/5: PPL CHECK")

            # Save the trained W_cons state so we can restore it after
            # temporarily reverting to the old checkpoint for PPL-before.
            trained_checkpoint: dict[str, torch.Tensor] = (
                self._dual_weights.save_cons_checkpoint()
            )

            # --- PPL before (old W_cons) ---
            self._dual_weights.restore_cons_checkpoint(cons_checkpoint)
            self._dual_weights.set_mode("target_inference")
            model.eval()
            ppl_before: float = _evaluate_ppl(
                model, benchmark_data, device=self._device,
            )
            result["ppl_before"] = ppl_before

            # --- PPL after (trained W_cons) ---
            self._dual_weights.restore_cons_checkpoint(trained_checkpoint)
            self._dual_weights.set_mode("target_inference")
            model.eval()
            ppl_after: float = _evaluate_ppl(
                model, benchmark_data, device=self._device,
            )
            result["ppl_after"] = ppl_after

            logger.info(
                "PPL check: before=%.4f, after=%.4f", ppl_before, ppl_after,
            )
            metrics.log({
                "sleep_engine/ppl_before": ppl_before,
                "sleep_engine/ppl_after": ppl_after,
            })

            # --- Rollback decision ---
            max_allowed: float = ppl_before * (
                1.0 + self._weights_config.epsilon_degrade
            )
            if ppl_after > max_allowed:
                logger.warning(
                    "PPL degraded beyond threshold (%.4f > %.4f) "
                    "— ROLLING BACK W_cons",
                    ppl_after, max_allowed,
                )
                self._dual_weights.restore_cons_checkpoint(cons_checkpoint)
                result["rolled_back"] = True
                metrics.log({"sleep_engine/rolled_back": 1})

                # All candidates treated as failed after rollback.
                validation_results_rb: dict[int, bool] = {
                    id(tag): False for tag in candidates
                }
                cleanup_result = cleanup_tags(
                    candidates, validation_results_rb, config,
                )
                result["n_consolidated"] = 0
                result["n_failed"] = len(cleanup_result["failed"])
                result["n_permanently_removed"] = len(
                    cleanup_result["permanently_removed"]
                )
                # On rollback: keep KV bank populated (memories may consolidate
                # in a future cycle). Re-enable injection for subsequent
                # wake/sleep cycles.
                self._set_kv_enabled(True)
                logger.info("=== Sleep cycle END (rolled back) ===")
                return result
        else:
            logger.info("Phase 4/5: PPL CHECK (skipped — no benchmark data)")

        # ================================================================
        # Phase 5 — VALIDATE & CLEANUP
        # ================================================================
        logger.info("Phase 5/5: VALIDATE & CLEANUP")

        # --- Compute old-model surprise for each candidate ---
        # Temporarily revert to pre-training W_cons.
        trained_checkpoint_v: dict[str, torch.Tensor] = (
            self._dual_weights.save_cons_checkpoint()
        )
        self._dual_weights.restore_cons_checkpoint(cons_checkpoint)
        self._dual_weights.set_mode("target_inference")
        model.eval()

        old_surprises: dict[int, float] = {}
        for tag in candidates:
            span_start, span_end, source_id = tag.ctx
            original_tokens = original_tokens_map.get(source_id)
            if original_tokens is not None:
                old_surprises[id(tag)] = compute_span_surprise(
                    model, original_tokens, span_start, span_end,
                    device=self._device,
                )
            else:
                old_surprises[id(tag)] = 0.0

        # --- Restore trained W_cons and validate ---
        self._dual_weights.restore_cons_checkpoint(trained_checkpoint_v)
        self._dual_weights.set_mode("target_inference")
        model.eval()

        validation_results_final: dict[int, bool] = {}
        for tag in candidates:
            _span_start, _span_end, source_id = tag.ctx
            original_tokens = original_tokens_map.get(source_id)
            old_surprise: float = old_surprises.get(id(tag), 0.0)

            passed = validate_consolidation(
                tag=tag,
                model_new=model,
                model_old_surprise=old_surprise,
                original_tokens=original_tokens,
                config=config,
                device=self._device,
            )
            validation_results_final[id(tag)] = passed

        # --- Cleanup: sort tags into outcome buckets ---
        cleanup_result = cleanup_tags(
            candidates, validation_results_final, config,
        )

        result["n_consolidated"] = len(cleanup_result["passed"])
        result["n_failed"] = len(cleanup_result["failed"])
        result["n_permanently_removed"] = len(
            cleanup_result["permanently_removed"]
        )

        logger.info(
            "Phase 5 complete: consolidated=%d, failed=%d, "
            "permanently_removed=%d",
            result["n_consolidated"],
            result["n_failed"],
            result["n_permanently_removed"],
        )
        metrics.log({
            "sleep_engine/n_consolidated": result["n_consolidated"],
            "sleep_engine/n_failed": result["n_failed"],
            "sleep_engine/n_permanently_removed": result[
                "n_permanently_removed"
            ],
        })
        metrics.log_summary({
            "sleep_engine/n_candidates": result["n_candidates"],
            "sleep_engine/n_replays_generated": result["n_replays_generated"],
            "sleep_engine/n_replays_accepted": result["n_replays_accepted"],
            "sleep_engine/n_consolidated": result["n_consolidated"],
            "sleep_engine/n_failed": result["n_failed"],
            "sleep_engine/n_permanently_removed": result[
                "n_permanently_removed"
            ],
            "sleep_engine/rolled_back": result["rolled_back"],
        })

        # --- KV memory: clear bank (knowledge has transferred to W_cons) ---
        # Only clear when at least one consolidation succeeded. If everything
        # failed, leave the bank populated for a subsequent cycle to retry.
        if result["n_consolidated"] > 0:
            self._clear_kv_bank_if_present()
        # Re-enable injection for the next cycle (no-op when bank is empty,
        # but ensures the toggle is in the expected state).
        self._set_kv_enabled(True)

        logger.info("=== Sleep cycle END ===")
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SleepEngine",
    "ReplaySample",
    "generate_replay",
    "quality_check",
    "sleep_train",
    "compute_fisher_diagonal",
    "compute_ewc_loss",
]
