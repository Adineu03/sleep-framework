"""Tests for sleep.tagging.tags — Tag dataclass, TagKeyProjection, create_tag factory."""

import math

import pytest
import torch

from sleep.config import TaggingConfig
from sleep.tagging.tags import Tag, TagKeyProjection, create_tag


# ---------------------------------------------------------------------------
# Tag creation with valid fields
# ---------------------------------------------------------------------------

class TestTagCreation:

    def test_tag_stores_all_core_fields(self):
        """Tag dataclass preserves every core field passed at construction."""
        k = torch.randn(128)
        tag = Tag(
            k=k,
            s=0.8,
            s0=0.8,
            s_reinforced=0.0,
            t0=42,
            e0=1.5,
            a=0,
            rho=0.0,
            ctx=(10, 20, "doc_1"),
        )
        assert torch.equal(tag.k, k)
        assert tag.s == 0.8
        assert tag.s0 == 0.8
        assert tag.s_reinforced == 0.0
        assert tag.t0 == 42
        assert tag.e0 == 1.5
        assert tag.a == 0
        assert tag.rho == 0.0
        assert tag.ctx == (10, 20, "doc_1")

    def test_tag_default_tracking_fields(self):
        """Implementation tracking fields have correct defaults."""
        tag = Tag(
            k=torch.zeros(64),
            s=0.5, s0=0.5, s_reinforced=0.0,
            t0=0, e0=1.0, a=0, rho=0.0,
            ctx=(0, 5, "src"),
        )
        assert tag.p == 0
        assert tag.S_score == 0.0
        assert tag.R == 0.0
        assert tag.R_last_update == 0
        assert tag.xref_count == 0
        assert tag.fail_count == 0
        assert tag.tag_type == "novelty"


# ---------------------------------------------------------------------------
# Tag serialization round-trip
# ---------------------------------------------------------------------------

class TestTagSerialization:

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict followed by from_dict reproduces an equivalent Tag."""
        k = torch.randn(128)
        original = Tag(
            k=k,
            s=0.75, s0=0.75, s_reinforced=0.1,
            t0=100, e0=2.3, a=3, rho=1.2,
            ctx=(5, 15, "doc_7"),
            p=1, S_score=0.42, R=0.9, R_last_update=95,
            xref_count=2, fail_count=1, tag_type="revision",
        )
        d = original.to_dict()
        restored = Tag.from_dict(d)

        assert torch.allclose(restored.k, original.k, atol=1e-6)
        assert restored.s == pytest.approx(original.s)
        assert restored.s0 == pytest.approx(original.s0)
        assert restored.s_reinforced == pytest.approx(original.s_reinforced)
        assert restored.t0 == original.t0
        assert restored.e0 == pytest.approx(original.e0)
        assert restored.a == original.a
        assert restored.rho == pytest.approx(original.rho)
        assert tuple(restored.ctx) == tuple(original.ctx)
        assert restored.p == original.p
        assert restored.S_score == pytest.approx(original.S_score)
        assert restored.R == pytest.approx(original.R)
        assert restored.R_last_update == original.R_last_update
        assert restored.xref_count == original.xref_count
        assert restored.fail_count == original.fail_count
        assert restored.tag_type == original.tag_type

    def test_from_dict_uses_specified_device(self):
        """from_dict places the key tensor on the requested device."""
        tag = Tag(
            k=torch.randn(64), s=0.5, s0=0.5, s_reinforced=0.0,
            t0=0, e0=1.0, a=0, rho=0.0, ctx=(0, 1, "x"),
        )
        restored = Tag.from_dict(tag.to_dict(), device="cpu")
        assert restored.k.device == torch.device("cpu")

    def test_to_dict_k_is_plain_list(self):
        """Serialized key is a plain Python list, not a tensor."""
        tag = Tag(
            k=torch.randn(32), s=0.5, s0=0.5, s_reinforced=0.0,
            t0=0, e0=1.0, a=0, rho=0.0, ctx=(0, 1, "x"),
        )
        d = tag.to_dict()
        assert isinstance(d["k"], list)
        assert all(isinstance(v, float) for v in d["k"])


# ---------------------------------------------------------------------------
# TagKeyProjection
# ---------------------------------------------------------------------------

class TestTagKeyProjection:

    @pytest.mark.parametrize("d_model,d_tag", [(768, 128), (1024, 64), (256, 256)])
    def test_output_shape(self, d_model, d_tag):
        """Projection maps (d_model,) -> (d_tag,) for a single vector."""
        proj = TagKeyProjection(d_model, d_tag)
        h_bar = torch.randn(d_model)
        k = proj(h_bar)
        assert k.shape == (d_tag,)

    def test_output_shape_batched(self):
        """Projection maps (B, d_model) -> (B, d_tag) for a batch."""
        proj = TagKeyProjection(768, 128)
        h_bar = torch.randn(4, 768)
        k = proj(h_bar)
        assert k.shape == (4, 128)


# ---------------------------------------------------------------------------
# create_tag factory (Q1.3)
# ---------------------------------------------------------------------------

class TestCreateTag:

    def test_create_tag_fields(self):
        """create_tag initialises all fields correctly per the spec."""
        config = TaggingConfig(d_tag=64, alpha_init=2.0)
        proj = TagKeyProjection(768, config.d_tag)

        h_bar = torch.randn(768)
        E_span = 1.5
        step = 10
        ctx = (3, 8, "doc_0")

        tag = create_tag(
            h_bar=h_bar,
            E_span=E_span,
            step=step,
            ctx=ctx,
            config=config,
            key_projection=proj,
        )

        # Key vector has correct shape
        assert tag.k.shape == (config.d_tag,)

        # s0 = sigmoid(alpha_init * E_span)
        expected_s0 = torch.sigmoid(torch.tensor(config.alpha_init * E_span)).item()
        assert tag.s0 == pytest.approx(expected_s0, rel=1e-5)
        assert tag.s == pytest.approx(expected_s0, rel=1e-5)

        # Zero-initialized counters
        assert tag.s_reinforced == 0.0
        assert tag.a == 0
        assert tag.rho == 0.0
        assert tag.p == 0

        # Timestamps and context
        assert tag.t0 == step
        assert tag.e0 == pytest.approx(E_span)
        assert tag.ctx == ctx
        assert tag.R_last_update == step
        assert tag.tag_type == "novelty"

    def test_create_tag_revision_type(self):
        """create_tag respects the tag_type parameter."""
        config = TaggingConfig(d_tag=32)
        proj = TagKeyProjection(128, config.d_tag)
        tag = create_tag(
            h_bar=torch.randn(128), E_span=1.0, step=1,
            ctx=(0, 5, "rev"), config=config, key_projection=proj,
            tag_type="revision",
        )
        assert tag.tag_type == "revision"

    def test_s0_increases_with_E_span(self):
        """Higher E_span should produce higher initial strength s0."""
        config = TaggingConfig(d_tag=32, alpha_init=2.0)
        proj = TagKeyProjection(128, config.d_tag)

        tag_low = create_tag(
            h_bar=torch.randn(128), E_span=0.5, step=1,
            ctx=(0, 1, "a"), config=config, key_projection=proj,
        )
        tag_high = create_tag(
            h_bar=torch.randn(128), E_span=3.0, step=1,
            ctx=(0, 1, "a"), config=config, key_projection=proj,
        )
        assert tag_high.s0 > tag_low.s0
