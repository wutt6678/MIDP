"""Shared naming/sanitization helpers for model-output artifacts.

R7 (Fix List for f59a9c1): the model-output directory name must be derived by
a single, shared sanitizer so the CLI, tests, and verification scripts never
disagree.  Previously ``final_verify.py`` hard-coded ``local--stub-vlm-v1``
while the CLI produced ``local_stub-vlm-v1``, causing the verifier to look in
the wrong directory.  Both now delegate to :func:`model_output_name`.
"""

from __future__ import annotations


def model_output_name(model_id: str) -> str:
    """Map an arbitrary model identifier to a filesystem-safe directory name.

    The transform replaces path/host separators (``/``, ``:``, ``\\``) with a
    single underscore and preserves every other character.  For example::

        local/stub-vlm-v1        -> local_stub-vlm-v1
        Qwen/Qwen2.5-VL-9B       -> Qwen_Qwen2.5-VL-9B

    The result is deterministic and reversible, so manifests and verifiers can
    reconstruct the directory from the configured ``model_id`` alone.
    """
    return (
        model_id
        .replace("/", "_")
        .replace(":", "_")
        .replace("\\", "_")
    )
