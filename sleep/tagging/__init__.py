"""
SLEEP Tagging Layer — main entry point.

Orchestrates the full tagging pipeline (Part 1, Q1.1-Q1.6):
  1. Per-token surprise computation via W_slow forward pass
  2. Adaptive z-score thresholding to flag surprising tokens
  3. Span segmentation (merge flagged regions)
  4. Tag creation (project hidden states, compute initial strength)
  5. Buffer management (capacity eviction, decay, garbage collection)

Usage::

    from sleep.tagging import TaggingLayer

    layer = TaggingLayer(model, config, model_params_billions=1.5)
    new_tags = layer.process_input(token_ids, source_id="doc_42")
    accessed = layer.process_query(query_ids)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import PreTrainedModel

from sleep.config import TaggingConfig
from sleep.utils.logging import get_logger, metrics
from sleep.tagging.surprise import compute_surprise, SurpriseResult
from sleep.tagging.threshold import AdaptiveThreshold

logger = get_logger("sleep.tagging")
from sleep.tagging.spans import segment_spans, Span
from sleep.tagging.tags import Tag, TagKeyProjection, create_tag
from sleep.tagging.buffer import TagBuffer

if TYPE_CHECKING:
    pass  # reserved for forward-reference-only imports


__all__ = [
    "TaggingLayer",
    "Tag",
    "TagKeyProjection",
    "TagBuffer",
    "AdaptiveThreshold",
    "SurpriseResult",
    "Span",
    "compute_surprise",
    "segment_spans",
    "create_tag",
]


class TaggingLayer:
    """Top-level tagging controller for the SLEEP system.

    Combines surprise computation, adaptive thresholding, span segmentation,
    tag creation, and buffer management into a single coherent pipeline.

    The underlying model (W_slow) is used for **inference only** — it is never
    modified by this module.

    Parameters
    ----------
    model:
        A HuggingFace ``PreTrainedModel`` (the W_slow base model).
    config:
        ``TaggingConfig`` containing all tagging hyperparameters.
    model_params_billions:
        Model size in billions of parameters, used to compute ``n_max``.
    tau_recency:
        Decay constant for recency-weighted utility (default 500).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        config: TaggingConfig,
        model_params_billions: float,
        tau_recency: int = 500,
    ) -> None:
        self._model: PreTrainedModel = model
        self._config: TaggingConfig = config
        self._model_params_billions: float = model_params_billions
        self._tau_recency: int = tau_recency

        # Derive model hidden size from the config object on the HF model.
        d_model: int = model.config.hidden_size

        # Key projection: h_bar (d_model,) -> k (d_tag,)
        self._key_projection = TagKeyProjection(d_model, config.d_tag)
        # Move projection to same device AND dtype as model (model may be in bfloat16/float16)
        model_param = next(model.parameters())
        self._key_projection = self._key_projection.to(device=model_param.device, dtype=model_param.dtype)

        # Adaptive threshold (EMA-based z-score flagging)
        self._threshold = AdaptiveThreshold(beta=config.beta, kappa=config.kappa)

        # Capacity: n_max = c_tag * model_params_billions
        self._n_max: int = int(config.c_tag * model_params_billions)

        # Tag buffer with capacity management
        self._buffer = TagBuffer(
            n_max=self._n_max,
            config=config,
            tau_recency=tau_recency,
        )

        # Global inference step counter
        self._step: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def step(self) -> int:
        """Current global inference step."""
        return self._step

    @property
    def active_tags(self) -> list[Tag]:
        """All tags currently alive in the buffer."""
        return self._buffer.tags

    @property
    def n_active(self) -> int:
        """Number of active tags in the buffer."""
        return self._buffer.n_active

    @property
    def occupancy(self) -> float:
        """Buffer occupancy as a fraction of n_max (0.0 to 1.0)."""
        return self._buffer.occupancy

    @property
    def threshold(self) -> AdaptiveThreshold:
        """The adaptive threshold instance (exposed for cold-start overrides)."""
        return self._threshold

    @property
    def key_projection(self) -> TagKeyProjection:
        """The tag key projection module."""
        return self._key_projection

    @property
    def buffer(self) -> TagBuffer:
        """The underlying tag buffer."""
        return self._buffer

    # ------------------------------------------------------------------
    # Main pipeline: process_input
    # ------------------------------------------------------------------

    def process_input(
        self,
        token_ids: torch.Tensor,
        source_id: str = "",
    ) -> list[Tag]:
        """Process an input sequence through the full tagging pipeline.

        Steps:
            1. Increment the global step counter.
            2. Forward pass through W_slow to get per-token surprise and hidden
               states (``compute_surprise``).
            3. Update adaptive threshold and flag surprising tokens.
            4. Segment flagged tokens into contiguous spans.
            5. Create a ``Tag`` for each span.
            6. Add new tags to the buffer (capacity eviction handled internally).

        Parameters
        ----------
        token_ids:
            1-D tensor of token IDs, shape ``(seq_len,)``.
        source_id:
            Optional identifier for the input source (e.g., document ID).

        Returns
        -------
        list[Tag]
            Newly created tags (already inserted into the buffer).
        """
        self._step += 1

        # Step 1-2: Forward pass + per-token surprise (inference only).
        device = next(self._model.parameters()).device
        result: SurpriseResult = compute_surprise(
            self._model, token_ids, device=str(device),
        )

        # Step 3: Adaptive thresholding — update running stats and flag tokens.
        flags: list[bool] = self._threshold.update_and_flag(result.surprises)

        # Step 4: Segment flagged tokens into spans.
        spans: list[Span] = segment_spans(
            flags=flags,
            surprises=result.surprises,
            hidden_states=result.hidden_states,
            running_mean=self._threshold.mu,
            gap_tolerance=self._config.gap_tolerance,
            min_span=self._config.min_span,
        )

        # Step 5: Create a tag for each span.
        new_tags: list[Tag] = []
        for span in spans:
            ctx = (span.start, span.end, source_id)
            tag = create_tag(
                h_bar=span.h_bar,
                E_span=span.E_span,
                step=self._step,
                ctx=ctx,
                config=self._config,
                key_projection=self._key_projection,
            )
            new_tags.append(tag)

        # Step 6: Add to buffer (buffer handles capacity eviction internally).
        self._buffer.add(new_tags)

        # Log
        if new_tags:
            logger.info(
                "step=%d | %d tags created from %d spans | buffer: %d/%d (%.1f%%)",
                self._step, len(new_tags), len(spans),
                self._buffer.n_active, self._n_max, self._buffer.occupancy * 100,
            )
            metrics.log({
                "tagging/n_tags_created": len(new_tags),
                "tagging/n_spans": len(spans),
                "tagging/mean_surprise": sum(result.surprises) / max(len(result.surprises), 1),
                "tagging/threshold_mu": self._threshold.mu,
                "tagging/threshold_sigma": self._threshold.sigma,
                "tagging/buffer_occupancy": self._buffer.occupancy,
            }, step=self._step)

        return new_tags

    # ------------------------------------------------------------------
    # Query processing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def process_query(self, token_ids: torch.Tensor) -> list[Tag]:
        """Process a query and return tags accessed by similarity.

        Steps:
            1. Forward pass through W_slow to obtain hidden states.
            2. Mean-pool hidden states into a single query representation.
            3. Project to tag key space via ``key_projection``.
            4. Delegate to the buffer for similarity-based access and
               reinforcement.

        Parameters
        ----------
        token_ids:
            1-D tensor of token IDs for the query, shape ``(seq_len,)``.

        Returns
        -------
        list[Tag]
            Tags whose key similarity to the query exceeds ``theta_access``.
        """
        device = next(self._model.parameters()).device
        token_ids = token_ids.to(device)

        # Forward pass (inference only).
        outputs = self._model(
            input_ids=token_ids.unsqueeze(0),
            output_hidden_states=True,
        )

        # Final-layer hidden states: (1, seq_len, d_model) -> (seq_len, d_model)
        final_hidden: torch.Tensor = outputs.hidden_states[-1][0]

        # Mean-pool across sequence length to get query representation.
        h_q: torch.Tensor = final_hidden.mean(dim=0)  # (d_model,)

        # Project to key space.
        query_key: torch.Tensor = self._key_projection(h_q)  # (d_tag,)

        # Buffer handles similarity matching, reinforcement, and access counting.
        accessed: list[Tag] = self._buffer.process_query(query_key, self._step)

        return accessed

    # ------------------------------------------------------------------
    # Periodic maintenance
    # ------------------------------------------------------------------

    def step_maintenance(self) -> None:
        """Run periodic maintenance on the tag buffer.

        Should be called every ``config.gc_interval`` steps.  Applies
        time-based decay to all tag strengths and garbage-collects tags
        whose strength has fallen below ``epsilon_gc``.
        """
        removed = self._buffer.decay_and_gc(self._step)
        if removed > 0:
            logger.debug("step=%d | GC removed %d tags | buffer: %d remaining", self._step, removed, self._buffer.n_active)