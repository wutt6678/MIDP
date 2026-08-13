"""Real-model 3-image × 3-attribute smoke test for Qwen3.5.

This test validates the scoring pipeline on real images with a real model.
It requires:
- Qwen3.5-9B model available (or via --model-id)
- 3 real face images via environment variable FIUBENCH_IMAGES (comma-separated)

The test performs 3 images × 3 attributes = 9 yes/no decisions and verifies:
- All log probabilities are finite
- Not all margins are identical (scoring is not collapsed)
- Not all p_positive ≈ 0.5 (model is discriminating)
- At least one attribute varies across images
- Free generation is nonblank
- Resolved model revision is present

Run with:
    FIUBENCH_IMAGES=/path/to/face1.jpg,/path/to/face2.jpg,/path/to/face3.jpg \\
    pytest tests/integration/test_qwen_real_model_smoke.py -v -s
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

# Skip all tests in this module unless explicitly enabled
pytestmark = pytest.mark.skipif(
    not os.environ.get("FIUBENCH_IMAGES"),
    reason="FIUBENCH_IMAGES not set; set to comma-separated paths to 3 face images",
)


def _get_test_images() -> list[Path]:
    """Parse FIUBENCH_IMAGES env var into a list of paths."""
    raw = os.environ.get("FIUBENCH_IMAGES", "")
    if not raw:
        return []
    paths = [Path(p.strip()) for p in raw.split(",") if p.strip()]
    return paths


def _load_image(path: Path):
    """Load a PIL image from path."""
    from PIL import Image

    return Image.open(path).convert("RGB")


def _load_backend(model_id: str = "Qwen/Qwen3.5-9B"):
    """Load the Qwen backend with real model."""
    from route_data.config import ModelConfig
    from route_data.models.qwen import QwenHFBackend

    # Minimal config for loading
    cfg = ModelConfig(
        model_id=model_id,
        backend="qwen_hf",
        revision=None,
        dtype="bfloat16",
        device_map="cuda:0",
        attn_implementation="sdpa",
        trust_remote_code=False,
        seed=42,
    )
    # Quantization disabled by default
    cfg.quantization.enabled = False
    # Generation defaults
    cfg.generation.do_sample = False
    cfg.generation.temperature = 0.0
    cfg.generation.max_new_tokens = 8

    backend = QwenHFBackend(cfg)
    return backend


def _binary_prompt(attribute: str) -> str:
    """Construct the standard binary question prompt."""
    return (
        f"Examine only the current image.\n"
        f"Does this person have {attribute}?\n"
        f"Answer exactly one word: Yes or No."
    )


def _p_positive(scored) -> float:
    """Compute p_positive from candidate scores (Yes/(Yes+No))."""
    logps = {
        cs.candidate: cs.log_probability
        for cs in (scored.candidate_scores or [])
    }
    log_p_yes = logps.get("Yes")
    log_p_no = logps.get("No")
    if log_p_yes is None or log_p_no is None:
        raise ValueError("missing Yes or No candidate score")
    # softmax in log space
    max_lp = max(log_p_yes, log_p_no)
    p_yes = math.exp(log_p_yes - max_lp) / (
        math.exp(log_p_yes - max_lp) + math.exp(log_p_no - max_lp)
    )
    return p_yes


class TestQwenRealModelSmoke:
    """Real-model 3-image × 3-attribute smoke test."""

    @pytest.fixture(scope="class")
    def backend(self):
        """Load the Qwen backend once for all tests in this class."""
        model_id = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3.5-9B")
        return _load_backend(model_id)

    @pytest.fixture(scope="class")
    def images(self):
        """Load the 3 test images."""
        paths = _get_test_images()
        assert len(paths) >= 3, (
            f"FIUBENCH_IMAGES must contain at least 3 comma-separated paths, "
            f"got {len(paths)}"
        )
        # Use exactly 3 images
        return [_load_image(p) for p in paths[:3]]

    @pytest.fixture(scope="class")
    def attributes(self):
        """Three CelebA attributes for smoke testing."""
        return ["Eyeglasses", "Smiling", "Wearing_Hat"]

    def test_resolved_model_revision_present(self, backend):
        """The backend must have a resolved model revision."""
        fingerprint = backend.fingerprint()
        assert "revision" in fingerprint
        assert fingerprint["revision"]
        assert fingerprint["revision"] != "unknown"

    def test_all_log_probabilities_finite(
        self, backend, images, attributes
    ):
        """All 9 yes/no decisions must have finite log probabilities."""
        for i, image in enumerate(images):
            for attr in attributes:
                prompt = _binary_prompt(attr)
                scored = backend.score_candidates(image, prompt, ["Yes", "No"])
                for cs in scored.candidate_scores:
                    assert math.isfinite(cs.log_probability), (
                        f"image {i}, attr={attr}, candidate={cs.candidate!r}: "
                        f"log_prob={cs.log_probability} is not finite"
                    )

    def test_not_all_margins_identical(self, backend, images, attributes):
        """Margins (log_p_yes - log_p_no) must not all be identical."""
        margins = []
        for image in images:
            for attr in attributes:
                prompt = _binary_prompt(attr)
                scored = backend.score_candidates(image, prompt, ["Yes", "No"])
                logps = {
                    cs.candidate: cs.log_probability
                    for cs in scored.candidate_scores
                }
                margin = logps["Yes"] - logps["No"]
                margins.append(margin)

        # At least some variation
        unique_margins = {round(m, 6) for m in margins}
        assert len(unique_margins) > 1, (
            f"all {len(margins)} margins are identical: {margins[0]:.6f}"
        )

    def test_not_all_p_positive_near_half(self, backend, images, attributes):
        """Not all p_positive values should be ≈ 0.5."""
        p_vals = []
        for image in images:
            for attr in attributes:
                prompt = _binary_prompt(attr)
                scored = backend.score_candidates(image, prompt, ["Yes", "No"])
                p = _p_positive(scored)
                p_vals.append(p)

        # At least one should be meaningfully different from 0.5
        near_half = [p for p in p_vals if abs(p - 0.5) < 0.01]
        assert len(near_half) < len(p_vals), (
            f"all {len(p_vals)} p_positive values are ≈ 0.5: {p_vals}"
        )

    def test_at_least_one_attribute_varies_across_images(
        self, backend, images, attributes
    ):
        """At least one attribute should have different scores across images."""
        variation_found = False
        for attr in attributes:
            p_vals = []
            for image in images:
                prompt = _binary_prompt(attr)
                scored = backend.score_candidates(image, prompt, ["Yes", "No"])
                p = _p_positive(scored)
                p_vals.append(p)
            spread = max(p_vals) - min(p_vals)
            if spread > 0.01:  # meaningful variation
                variation_found = True
                break

        assert variation_found, (
            "no attribute varies across images; all scores are image-invariant"
        )

    def test_free_generation_nonblank(self, backend, images, attributes):
        """Free generation should produce nonblank text."""
        all_blank = True
        for image in images:
            for attr in attributes:
                prompt = _binary_prompt(attr)
                gen = backend.generate(image, prompt)
                if gen.text and gen.text.strip():
                    all_blank = False
                    break
            if not all_blank:
                break

        assert not all_blank, "all free generation outputs are blank"

    def test_scoring_debug_metadata_present(self, backend, images, attributes):
        """scoring_debug metadata should be present and well-formed."""
        image = images[0]
        attr = attributes[0]
        prompt = _binary_prompt(attr)
        scored = backend.score_candidates(image, prompt, ["Yes", "No"])

        debug = scored.metadata.get("scoring_debug")
        assert debug is not None, "scoring_debug metadata missing"
        assert len(debug) == 2, f"expected 2 debug entries, got {len(debug)}"

        for entry in debug:
            assert "candidate" in entry
            assert "candidate_token_ids" in entry
            assert "prefix_length" in entry
            assert "full_length" in entry
            assert "scored_positions" in entry
            assert entry["prefix_length"] > 0
            assert entry["full_length"] == entry["prefix_length"] + len(
                entry["candidate_token_ids"]
            )
            assert len(entry["candidate_token_ids"]) > 0
