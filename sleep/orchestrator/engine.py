"""
SLEEP system orchestrator engine.

The ``SLEEPEngine`` is the top-level class that ties all SLEEP modules
together: tagging, PRP allocation, dual weights, sleep consolidation,
and cold-start management.  Users interact with this class exclusively.

Usage::

    from sleep.orchestrator import SLEEPEngine
    from sleep.config import SLEEPConfig

    engine = SLEEPEngine(model, tokenizer, SLEEPConfig())
    result = engine.process_input("The Q3 revenue was $4.2M")
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import torch
from transformers import PreTrainedModel

from sleep.config import SLEEPConfig
from sleep.orchestrator.cold_start import ColdStartManager
from sleep.orchestrator.state import Phase, SystemState
from sleep.orchestrator.triggers import should_sleep
from sleep.tagging import TaggingLayer, Tag
from sleep.prp import PRPSystem
from sleep.weights import DualWeightSystem
from sleep.sleep_engine import SleepEngine
from sleep.utils.logging import get_logger, metrics

logger = get_logger("sleep.orchestrator.engine")


def _count_model_params_billions(model: PreTrainedModel) -> float:
    """Count total parameters in billions."""
    total = sum(p.numel() for p in model.parameters())
    return total / 1e9


class SLEEPEngine:
    """The complete SLEEP system orchestrator.

    This is the top-level class that users interact with.  It manages the full
    wake/sleep cycle: processing inputs, managing tags, scoring PRPs, triggering
    sleep, running consolidation, and tracking state.

    Args:
        model:     A HuggingFace ``PreTrainedModel`` (causal LM).
        tokenizer: The matching HuggingFace tokenizer.
        config:    Complete ``SLEEPConfig`` with all sub-configs.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Any,
        config: SLEEPConfig,
    ) -> None:
        self._config = config
        self._tokenizer = tokenizer

        # 1. Compute model size and store in config
        model_params_b = _count_model_params_billions(model)
        config.model_params_billions = model_params_b
        logger.info(
            "Initializing SLEEPEngine | model=%.3fB params | device=%s",
            model_params_b, config.device,
        )

        # 2. Create DualWeightSystem (wraps model with LoRA adapters)
        self._dual_weights = DualWeightSystem(model, config.weights)

        # 3. Create TaggingLayer (uses base model for surprise computation)
        #    The tagging layer needs the underlying base model (pre-LoRA) for
        #    stable surprise measurement.  DualWeightSystem stores it as .model,
        #    which is a PeftModel.  We pass the full PeftModel — TaggingLayer
        #    runs inference-only forward passes through it.
        self._tagging = TaggingLayer(
            model=self._dual_weights.model,
            config=config.tagging,
            model_params_billions=model_params_b,
            tau_recency=config.prp.tau_recency,
        )

        # 4. Create PRPSystem
        prp_budget = config.prp_budget
        self._prp = PRPSystem(
            config=config.prp,
            budget=prp_budget,
            revision_bonus=config.revision.w_revision_bonus,
        )

        # 5. Create SleepEngine
        self._sleep_engine = SleepEngine(
            dual_weights=self._dual_weights,
            tokenizer=tokenizer,
            sleep_config=config.sleep,
            weights_config=config.weights,
            device=config.device,
        )

        # 6. Create ColdStartManager
        self._cold_start = ColdStartManager(
            config=config.cold_start,
            normal_kappa=config.tagging.kappa,
        )

        # 7. Initialize SystemState
        self._state = SystemState()

        # Token cache for replay generation: source_id -> token_ids
        self._token_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._token_cache_max: int = 10000

        # Timing for idle trigger
        self._last_interaction_time: float = time.monotonic()

        # PRP allocation stats cache (for trigger evaluation)
        self._last_prp_stats: dict = {"budget_utilization": 0.0}

        logger.info(
            "SLEEPEngine ready | n_max_tags=%d | prp_budget=%d | cold_start_burnin=%d",
            config.n_max_tags, prp_budget, config.cold_start.n_burnin,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> SystemState:
        """Current system state."""
        return self._state

    @property
    def stats(self) -> dict:
        """Current system statistics."""
        return {
            **self._state.to_dict(),
            "buffer_occupancy": self._tagging.occupancy,
            "n_active_tags": self._tagging.n_active,
            "n_max_tags": self._tagging._n_max,
            "prp_budget": self._prp.budget,
            "prp_budget_utilization": self._last_prp_stats.get("budget_utilization", 0.0),
            "cold_start_mature": self._cold_start.is_mature,
            "cold_start_interactions": self._cold_start.interaction_count,
            "token_cache_size": len(self._token_cache),
        }

    @property
    def tagging(self) -> TaggingLayer:
        """The tagging layer instance."""
        return self._tagging

    @property
    def prp(self) -> PRPSystem:
        """The PRP system instance."""
        return self._prp

    @property
    def dual_weights(self) -> DualWeightSystem:
        """The dual weight system instance."""
        return self._dual_weights

    @property
    def sleep_engine(self) -> SleepEngine:
        """The sleep engine instance."""
        return self._sleep_engine

    @property
    def cold_start(self) -> ColdStartManager:
        """The cold-start manager instance."""
        return self._cold_start

    # ------------------------------------------------------------------
    # Wake phase: process_input
    # ------------------------------------------------------------------

    def process_input(self, text: str, source_id: str = "") -> dict:
        """Process a text input during wake phase.

        Steps:
            1. Tokenize the input text.
            2. Run through TaggingLayer to create tags for surprising spans.
            3. If any span exceeds ``kappa_wfast``, perform a W_fast online update.
            4. Periodically run PRP scoring/allocation.
            5. Periodically run tag decay and garbage collection.
            6. Check sleep triggers.
            7. If a trigger fires, run a full sleep cycle.

        Args:
            text:      Input text string.
            source_id: Optional identifier for the input source.  Defaults to
                       ``"input_{step}"``.

        Returns:
            A dict with processing results::

                {
                    "n_tags_created": int,
                    "n_accessed": int,
                    "wfast_updated": bool,
                    "sleep_triggered": bool,
                    "sleep_results": dict | None,
                    "state": SystemState,
                }
        """
        result: dict[str, Any] = {
            "n_tags_created": 0,
            "n_accessed": 0,
            "wfast_updated": False,
            "sleep_triggered": False,
            "sleep_results": None,
            "state": self._state,
        }

        try:
            self._last_interaction_time = time.monotonic()

            # Ensure wake mode
            self._dual_weights.set_mode("wake_inference")

            # Update state
            self._state.step += 1
            self._state.steps_since_last_sleep += 1
            self._state.interaction_count += 1

            if not source_id:
                source_id = f"input_{self._state.step}"

            # --- Cold start: adjust kappa for tagging ---
            self._cold_start.record_interaction()
            effective_kappa = self._cold_start.get_effective_kappa(
                self._cold_start.interaction_count,
            )
            self._tagging.threshold.kappa = effective_kappa

            # 1. Tokenize
            token_ids = self._tokenize(text)

            # 2. Cache tokens for replay generation
            self._cache_tokens(source_id, token_ids)

            # 3. Run through TaggingLayer -> create tags
            new_tags: list[Tag] = self._tagging.process_input(token_ids, source_id)
            result["n_tags_created"] = len(new_tags)
            self._state.total_tags_created += len(new_tags)

            # 4. W_fast update for highly surprising spans
            wfast_updated = self._maybe_update_wfast(new_tags, token_ids)
            result["wfast_updated"] = wfast_updated

            # 5. Periodic PRP scoring/allocation
            step = self._state.step
            if step % self._config.prp.allocation_interval == 0:
                self._run_prp_update(step)

            # 6. Periodic tag decay and GC
            if step % self._config.tagging.gc_interval == 0:
                self._tagging.step_maintenance()

            # 7. Check sleep triggers
            idle_seconds = time.monotonic() - self._last_interaction_time
            trigger, reason = should_sleep(
                state=self._state,
                buffer_occupancy=self._tagging.occupancy,
                budget_utilization=self._last_prp_stats.get("budget_utilization", 0.0),
                idle_seconds=idle_seconds,
                config=self._config.sleep,
            )

            if trigger:
                logger.info("Sleep triggered during process_input: %s", reason)
                sleep_results = self._run_sleep_cycle()
                result["sleep_triggered"] = True
                result["sleep_results"] = sleep_results

            result["state"] = self._state

            # Log metrics
            metrics.log({
                "orchestrator/n_tags_created": result["n_tags_created"],
                "orchestrator/wfast_updated": int(result["wfast_updated"]),
                "orchestrator/step": self._state.step,
                "orchestrator/buffer_occupancy": self._tagging.occupancy,
            }, step=self._state.step)

        except Exception:
            logger.exception("Error in process_input (step=%d)", self._state.step)
            # Return partial result rather than crashing
            result["state"] = self._state

        return result

    # ------------------------------------------------------------------
    # Wake phase: process_query
    # ------------------------------------------------------------------

    def process_query(self, text: str) -> dict:
        """Process a query for tag access and reinforcement tracking.

        Runs the query through the tagging layer's similarity-based
        retrieval pipeline.  Accessed tags get reinforced (strength boost).

        Args:
            text: Query text string.

        Returns:
            A dict with ``{"n_accessed": int, "accessed_tags": list[Tag]}``.
        """
        result: dict[str, Any] = {"n_accessed": 0, "accessed_tags": []}

        try:
            self._last_interaction_time = time.monotonic()
            self._dual_weights.set_mode("wake_inference")

            token_ids = self._tokenize(text)
            accessed_tags: list[Tag] = self._tagging.process_query(token_ids)
            result["n_accessed"] = len(accessed_tags)
            result["accessed_tags"] = accessed_tags

            if accessed_tags:
                logger.debug(
                    "Query accessed %d tags", len(accessed_tags),
                )

        except Exception:
            logger.exception("Error in process_query")

        return result

    # ------------------------------------------------------------------
    # Sleep phase: force and internal
    # ------------------------------------------------------------------

    def force_sleep(self) -> dict:
        """Force a sleep cycle regardless of trigger conditions.

        Returns:
            Sleep cycle results dict (see :meth:`_run_sleep_cycle`).
        """
        logger.info("Force-sleep requested")
        return self._run_sleep_cycle()

    def set_baseline(self, calibration_data: list[torch.Tensor]) -> float:
        """Set the baseline surprise for the sleep engine.

        Must be called before any sleep cycle can run.  Pass a list of
        tokenized calibration sequences (general text the model already
        knows well).

        Args:
            calibration_data: List of 1-D token-ID tensors.

        Returns:
            The computed ``mu_surprise`` value.
        """
        return self._sleep_engine.set_baseline(calibration_data)

    def _run_sleep_cycle(self) -> dict:
        """Execute one full sleep cycle.

        Steps:
            1. Set phase to SLEEP.
            2. Run a PRP update to ensure scores are current.
            3. Get consolidation candidates from PRP.
            4. Gather original tokens for replay from the cache.
            5. Call ``SleepEngine.run_cycle()``.
            6. Update system state with results.
            7. Set phase back to WAKE.

        Returns:
            Sleep cycle results dict from ``SleepEngine.run_cycle()``.
        """
        result: dict = {
            "n_candidates": 0,
            "n_consolidated": 0,
            "n_failed": 0,
            "error": None,
        }

        try:
            self._state.phase = Phase.SLEEP
            logger.info(
                "=== SLEEP CYCLE %d START (step=%d) ===",
                self._state.sleep_cycle_count + 1, self._state.step,
            )

            # 1. Ensure PRP scores are current
            self._run_prp_update(self._state.step, force_crossref=True)

            # 2. Get consolidation candidates
            candidates = self._prp.get_consolidation_candidates(
                self._tagging.active_tags,
            )
            result["n_candidates"] = len(candidates)

            if not candidates:
                logger.info("No consolidation candidates — skipping sleep cycle")
                self._state.phase = Phase.WAKE
                return result

            # 3. Build original_tokens_map from cache
            original_tokens_map: dict[str, torch.Tensor] = {}
            for tag in candidates:
                _start, _end, source_id = tag.ctx
                if source_id in self._token_cache:
                    original_tokens_map[source_id] = self._token_cache[source_id]

            # 4. Run the sleep engine cycle
            cycle_result = self._sleep_engine.run_cycle(
                candidates=candidates,
                original_tokens_map=original_tokens_map,
                key_projection=self._tagging.key_projection,
            )

            # 5. Update system state
            self._state.sleep_cycle_count += 1
            self._state.steps_since_last_sleep = 0
            self._state.last_sleep_step = self._state.step
            self._state.total_memories_consolidated += cycle_result.get(
                "n_consolidated", 0,
            )
            self._state.total_memories_failed += cycle_result.get(
                "n_failed", 0,
            )

            result.update(cycle_result)

            logger.info(
                "=== SLEEP CYCLE %d END | consolidated=%d, failed=%d ===",
                self._state.sleep_cycle_count,
                cycle_result.get("n_consolidated", 0),
                cycle_result.get("n_failed", 0),
            )

            metrics.log_summary({
                "orchestrator/sleep_cycle": self._state.sleep_cycle_count,
                "orchestrator/total_consolidated": self._state.total_memories_consolidated,
                "orchestrator/total_failed": self._state.total_memories_failed,
            })

        except Exception as e:
            logger.exception("Error during sleep cycle")
            result["error"] = str(e)

        finally:
            self._state.phase = Phase.WAKE
            self._dual_weights.set_mode("wake_inference")

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text to a 1-D tensor of token IDs on the correct device."""
        encoding = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids: torch.Tensor = encoding["input_ids"].squeeze(0)  # (seq_len,)
        return token_ids.to(self._config.device)

    def _cache_tokens(self, source_id: str, token_ids: torch.Tensor) -> None:
        """Store token_ids in the bounded cache for later replay generation.

        When the cache is full, evicts entries whose tags have all decayed
        (no longer present in the buffer).  If no such entries exist, evicts
        the oldest entry.
        """
        if source_id in self._token_cache:
            # Move to end (most recently used)
            self._token_cache.move_to_end(source_id)
            self._token_cache[source_id] = token_ids.detach().cpu()
            return

        # Evict if at capacity
        while len(self._token_cache) >= self._token_cache_max:
            evicted = False
            # First pass: evict entries with no surviving tags
            active_sources = {t.ctx[2] for t in self._tagging.active_tags}
            for cached_id in list(self._token_cache.keys()):
                if cached_id not in active_sources:
                    del self._token_cache[cached_id]
                    evicted = True
                    break

            if not evicted:
                # All cached entries still have active tags — evict oldest
                self._token_cache.popitem(last=False)

        self._token_cache[source_id] = token_ids.detach().cpu()

    def _maybe_update_wfast(self, new_tags: list[Tag], token_ids: torch.Tensor) -> bool:
        """Update W_fast if any new tag has surprise above kappa_wfast threshold.

        Returns True if at least one W_fast update was performed.
        """
        if not new_tags:
            return False

        kappa_wfast = self._config.tagging.kappa_wfast
        threshold_mu = self._tagging.threshold.mu
        threshold_sigma = self._tagging.threshold.sigma

        # W_fast threshold: mu + kappa_wfast * sigma
        wfast_threshold = threshold_mu + kappa_wfast * max(threshold_sigma, 1e-6)

        updated = False
        for tag in new_tags:
            if tag.e0 > wfast_threshold:
                try:
                    span_start, span_end, _source_id = tag.ctx
                    self._dual_weights.update_fast_weights(
                        token_ids=token_ids,
                        span_start=span_start,
                        span_end=span_end,
                        E_span=tag.e0,
                        device=self._config.device,
                    )
                    updated = True
                    self._state.total_wfast_updates += 1
                except Exception:
                    logger.exception(
                        "W_fast update failed for tag at span [%d:%d]",
                        tag.ctx[0], tag.ctx[1],
                    )

        return updated

    def _run_prp_update(self, step: int, force_crossref: bool = False) -> None:
        """Run a PRP scoring and allocation update.

        Applies cold-start budget scaling if the system is not yet mature.
        """
        try:
            # Apply cold-start budget scaling
            budget_scale = self._cold_start.get_budget_scale(
                self._cold_start.interaction_count,
            )
            effective_budget = max(1, int(self._prp.budget * budget_scale))

            # Temporarily adjust budget for this update
            original_budget = self._prp._budget
            self._prp._budget = effective_budget

            stats = self._prp.update(
                tags=self._tagging.active_tags,
                current_step=step,
                force_crossref=force_crossref,
            )
            self._last_prp_stats = stats

            # Restore original budget
            self._prp._budget = original_budget

        except Exception:
            logger.exception("PRP update failed at step %d", step)
