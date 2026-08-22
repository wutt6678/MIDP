"""Adapter registry for trainable VLM models.

Maps adapter *family* names (e.g. ``"qwen35"``) to adapter classes, and
*model keys* (e.g. ``"qwen35_9b"``) to families.  The YAML profile is the
single source of truth — adapters are constructed from profiles, not from
hard-coded internal defaults.

Architecture
------------
::

    YAML profile  ──►  load_profile_from_yaml()  ──►  ModelFamilyProfile
                                                          │
    profile.adapter_name ──►  _ADAPTER_FAMILIES[family]  ──►  AdapterClass(profile)
    profile.key ──► _MODEL_KEY_TO_FAMILY[key] ──► family ──┘

There must be exactly one effective experimental profile per adapter
instance: ``adapter.profile is profile``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .base import ModelFamilyProfile, TrainableVLMAdapter

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Adapter family registry
# ------------------------------------------------------------------ #

# family name → adapter class (accepts profile in __init__)
_ADAPTER_FAMILIES: dict[str, type[TrainableVLMAdapter]] = {}

# model key → adapter family name
_MODEL_KEY_TO_FAMILY: dict[str, str] = {}

# model key → cached adapter instance
_ADAPTER_CACHE: dict[str, TrainableVLMAdapter] = {}


def register_adapter_family(name: str) -> Callable:
    """Decorator to register an adapter *family* class.

    Usage::

        @register_adapter_family("qwen35")
        class Qwen35Adapter(HuggingFaceChatAdapter):
            def __init__(self, profile: ModelFamilyProfile):
                ...
    """
    def decorator(cls: type[TrainableVLMAdapter]) -> type[TrainableVLMAdapter]:
        if name in _ADAPTER_FAMILIES:
            raise ValueError(
                f"Adapter family already registered: {name!r}"
            )
        _ADAPTER_FAMILIES[name] = cls
        return cls
    return decorator


def register_model_key(key: str, family: str) -> None:
    """Map a model key to an adapter family.

    Multiple model keys can share the same family (e.g. ``qwen35_9b``
    and ``qwen35_4b`` both use the ``qwen35`` family).
    """
    _MODEL_KEY_TO_FAMILY[key] = family


def create_adapter(
    key: str,
    profile: ModelFamilyProfile | None = None,
) -> TrainableVLMAdapter:
    """Create an adapter for the given model key.

    Parameters
    ----------
    key:
        Model key (e.g. ``"qwen35_9b"``).
    profile:
        The YAML-loaded profile.  **Required** for research-mode runs.
        The adapter's ``.profile`` will be this exact object.

    Returns
    -------
    adapter : TrainableVLMAdapter
        Cached adapter instance with ``adapter.profile is profile``.
    """
    if profile is None:
        raise ValueError(
            f"create_adapter({key!r}): profile is required. "
            f"Load a YAML profile first."
        )

    cache_key = f"{key}:{id(profile)}"

    if cache_key not in _ADAPTER_CACHE:
        _ensure_builtin_adapters_loaded()

        if key not in _MODEL_KEY_TO_FAMILY:
            raise KeyError(
                f"Unknown model key: {key!r}. "
                f"Available: {sorted(_MODEL_KEY_TO_FAMILY)}"
            )

        family_name = _MODEL_KEY_TO_FAMILY[key]
        if family_name not in _ADAPTER_FAMILIES:
            raise KeyError(
                f"Adapter family {family_name!r} not registered "
                f"for model key {key!r}"
            )

        # P0-10: Profile adapter_name must match registered family
        if profile.adapter_name != family_name:
            raise ValueError(
                f"Profile {key!r} has adapter_name={profile.adapter_name!r} "
                f"but model key is registered to family {family_name!r}. "
                f"Fix the YAML profile."
            )

        family_cls = _ADAPTER_FAMILIES[family_name]
        adapter = family_cls(profile)

        # Invariant: adapter.profile is the provided profile
        if adapter.profile is not profile:
            raise RuntimeError(
                f"Adapter {key!r} did not bind the provided profile. "
                f"adapter.profile is {adapter.profile!r}, expected {profile!r}"
            )

        _ADAPTER_CACHE[cache_key] = adapter

    return _ADAPTER_CACHE[cache_key]


def available_adapters() -> list[str]:
    """Return sorted list of registered model keys."""
    _ensure_builtin_adapters_loaded()
    return sorted(_MODEL_KEY_TO_FAMILY.keys())


def adapter_families() -> list[str]:
    """Return sorted list of registered adapter family names."""
    _ensure_builtin_adapters_loaded()
    return sorted(_ADAPTER_FAMILIES.keys())


# ------------------------------------------------------------------ #
# Profile YAML loading
# ------------------------------------------------------------------ #

# Pattern for unresolved placeholders like <PIN_EXACT_HF_COMMIT_SHA>
_PLACEHOLDER_RE = re.compile(r"^<.*>$")

# Full 40-char hex SHA pattern
_FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def load_profile_from_yaml(path: str | Path) -> ModelFamilyProfile:
    """Load a :class:`ModelFamilyProfile` from a YAML configuration file.

    The YAML schema mirrors the ``ModelFamilyProfile`` dataclass fields
    with nested sections: ``model``, ``candidate_protocol``, ``lora``,
    ``structural``, ``compatibility``, ``access``.

    Performs basic type validation before constructing the dataclass.
    """
    import yaml

    with open(path) as fh:
        raw_bytes = fh.read()

    data = yaml.safe_load(raw_bytes)

    model = data["model"]
    candidate = data.get("candidate_protocol", {})
    lora = data.get("lora", {})
    structural = data.get("structural", {})
    compat = data.get("compatibility", {})
    access = data.get("access", {})

    # --- Type validation ---
    _validate_profile_types(lora, structural, access, data)

    # --- Structural fields ---
    raw_layers = structural.get("r2mu_candidate_layers", [])
    if not isinstance(raw_layers, list):
        raise TypeError(
            f"r2mu_candidate_layers must be a list[int], "
            f"got {type(raw_layers).__name__}: {raw_layers!r}"
        )
    if not all(isinstance(v, int) for v in raw_layers):
        raise ValueError(
            f"r2mu_candidate_layers must be list[int], "
            f"got {[type(v).__name__ for v in raw_layers]}"
        )

    raw_leaf_names = lora.get("target_leaf_names", [])
    if not isinstance(raw_leaf_names, list):
        raise TypeError(
            f"lora.target_leaf_names must be a list[str], "
            f"got {type(raw_leaf_names).__name__}"
        )

    # Support both nested compatibility.constraints format and legacy flat format.
    constraints = compat.get("constraints", {}) if "constraints" in compat else compat
    min_transformers = constraints.get("min_transformers")
    max_transformers_exclusive = constraints.get("max_transformers_exclusive")
    tested_transformers = (
        compat.get("tested_environment", {}).get("transformers", "")
        if "tested_environment" in compat
        else compat.get("tested_transformers", "")
    )

    return ModelFamilyProfile(
        key=data["key"],
        model_id=model["id"],
        revision=model["revision"],
        processor_id=model.get("processor_id", model["id"]),
        processor_revision=model.get("processor_revision", model["revision"]),
        adapter_name=model.get("adapter", data["key"]),
        trust_remote_code=_safe_bool(model.get("trust_remote_code", False)),
        dtype=model.get("dtype", "bfloat16"),
        attn_implementation=model.get("attn_implementation", "sdpa"),
        candidate_positive=candidate.get("positive", "Yes"),
        candidate_negative=candidate.get("negative", "No"),
        lora_rank=_safe_int(lora.get("rank", 8), "lora.rank"),
        lora_alpha=_safe_int(lora.get("alpha", 16), "lora.alpha"),
        lora_dropout=_safe_float(lora.get("dropout", 0.05), "lora.dropout"),
        lora_scope=lora.get("scope", "language_attention_only"),
        lora_target_leaf_names=tuple(raw_leaf_names),
        lora_scope_regex=lora.get("scope_regex", ""),
        lora_expected_target_modules=_safe_int(
            lora.get("expected_target_modules", 0),
            "lora.expected_target_modules",
        ) if "expected_target_modules" in lora else 0,
        r2mu_candidate_layers=tuple(raw_layers),
        r2mu_n_select_layers=_safe_int(
            structural.get("r2mu_n_select_layers", 0),
            "structural.r2mu_n_select_layers",
        ),
        # P0-5: Structural metadata (optional, defaults to empty/zero)
        language_layer_path=structural.get("language_layer_path", ""),
        language_hidden_size=_safe_int(
            structural.get("language_hidden_size", 0),
            "structural.language_hidden_size",
        ),
        intermediate_size=_safe_int(
            structural.get("intermediate_size", 0),
            "structural.intermediate_size",
        ),
        num_language_layers=_safe_int(
            structural.get("num_language_layers", 0),
            "structural.num_language_layers",
        ),
        supports_prompting=_safe_bool(data.get("supports_prompting", True)),
        supports_candidate_margin=_safe_bool(
            data.get("supports_candidate_margin", True)
        ),
        supports_ga=_safe_bool(data.get("supports_ga", True)),
        supports_gd=_safe_bool(data.get("supports_gd", True)),
        supports_kl=_safe_bool(data.get("supports_kl", True)),
        supports_npo=_safe_bool(data.get("supports_npo", True)),
        supports_mmunlearner=_safe_bool(data.get("supports_mmunlearner", True)),
        supports_manu=_safe_bool(data.get("supports_manu", True)),
        supports_r2mu=_safe_bool(data.get("supports_r2mu", True)),
        min_transformers_version=min_transformers,
        max_transformers_version_exclusive=max_transformers_exclusive,
        tested_transformers_version=tested_transformers,
        requires_hf_auth=_safe_bool(access.get("requires_hf_auth", False)),
    )


def compute_profile_sha256(path: str | Path) -> str:
    """Compute SHA-256 over the *scientific execution* fields of a profile.

    Mutable capability/status metadata (``supports_*``, ``status``,
    ``access``) and comments/whitespace are **excluded** so that
    promoting a model from PENDING to PASS does not invalidate its
    frozen baseline binding.

    P1-PHI-01: For ``compatibility``, only ``constraints`` is hashed
    (enforceable version ranges). ``tested_environment`` is descriptive
    metadata and does not affect execution semantics.

    Included sections: ``key``, ``model``, ``candidate_protocol``,
    ``lora``, ``structural``, ``compatibility.constraints``.
    """
    import yaml

    with open(path) as fh:
        data = yaml.safe_load(fh)

    # Keep only scientific execution fields.
    _SCIENTIFIC_KEYS = [
        "key", "model", "candidate_protocol", "lora", "structural",
    ]
    canonical = {k: data[k] for k in _SCIENTIFIC_KEYS if k in data}

    # P1-PHI-01: For compatibility, only hash constraints (not tested_environment)
    compat = data.get("compatibility", {})
    if compat:
        # Support both old flat format and new nested format
        if "constraints" in compat:
            canonical["compatibility"] = {"constraints": compat["constraints"]}
        else:
            # Old format: extract only constraint fields
            constraint_keys = ["min_transformers", "max_transformers_exclusive",
                               "min_torch", "max_torch_exclusive",
                               "min_peft", "max_peft_exclusive"]
            constraints = {k: compat[k] for k in constraint_keys if k in compat}
            if constraints:
                canonical["compatibility"] = {"constraints": constraints}

    # Deterministic canonical serialisation.
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# ------------------------------------------------------------------ #
# Research-profile validation (P0-E)
# ------------------------------------------------------------------ #

def validate_research_profile(profile: ModelFamilyProfile) -> list[str]:
    """Validate a profile for research-mode use.

    Returns a list of error messages.  Empty list means the profile is
    valid for research use.

    Checks
    ------
    - Revisions are exact 40-hex SHA (not placeholders, not branch names)
    - No unresolved ``<...>`` placeholders in critical fields
    - Capability-consistent fields (e.g. supports_ga → revision pinned)
    """
    errors: list[str] = []

    # --- Revision validation ---
    for name, val in [
        ("revision", profile.revision),
        ("processor_revision", profile.processor_revision),
    ]:
        if not val:
            errors.append(f"{name} is empty")
        elif _PLACEHOLDER_RE.match(val):
            errors.append(
                f"{name} contains unresolved placeholder: {val!r}"
            )
        elif not _FULL_SHA_RE.fullmatch(val):
            errors.append(
                f"{name}={val!r} is not a 40-char hex SHA"
            )

    # --- Placeholder check on other critical fields ---
    if profile.lora_scope_regex and _PLACEHOLDER_RE.match(
        profile.lora_scope_regex
    ):
        errors.append(
            f"lora_scope_regex contains unresolved placeholder: "
            f"{profile.lora_scope_regex!r}"
        )

    # --- Capability-consistent Validation ---
    any_method_supported = any([
        profile.supports_prompting,
        profile.supports_candidate_margin,
        profile.supports_ga,
        profile.supports_gd,
        profile.supports_kl,
        profile.supports_npo,
        profile.supports_mmunlearner,
        profile.supports_manu,
        profile.supports_r2mu,
    ])
    
    if any_method_supported:
        # Must have pinned revisions
        if not _FULL_SHA_RE.fullmatch(profile.revision):
            errors.append(
                "Methods enabled but revision is not a pinned SHA"
            )
        if not _FULL_SHA_RE.fullmatch(profile.processor_revision):
            errors.append(
                "Methods enabled but processor_revision is not a pinned SHA"
            )
        if not profile.lora_scope_regex or _PLACEHOLDER_RE.match(
            profile.lora_scope_regex
        ):
            errors.append(
                "Methods enabled but lora_scope_regex is unresolved"
            )
    
    # Capability-specific validation
    _loralike_methods = [
        profile.supports_ga,
        profile.supports_gd,
        profile.supports_kl,
        profile.supports_npo,
        profile.supports_mmunlearner,
    ]
    if any(_loralike_methods) and not profile.lora_target_leaf_names:
        errors.append(
            "LoRA-requiring methods enabled but lora_target_leaf_names is empty"
        )
    
    if profile.supports_manu and not profile.r2mu_candidate_layers:
        # MANU needs structural layer info
        errors.append(
            "supports_manu=true but r2mu_candidate_layers is empty"
        )
    
    if profile.supports_r2mu and not profile.r2mu_candidate_layers:
        errors.append(
            "supports_r2mu=true but r2mu_candidate_layers is empty"
        )
    if (
        profile.supports_r2mu
        and profile.r2mu_candidate_layers
        and profile.r2mu_n_select_layers > len(profile.r2mu_candidate_layers)
    ):
        errors.append(
            f"r2mu_n_select_layers ({profile.r2mu_n_select_layers}) "
            f"> len(r2mu_candidate_layers) ({len(profile.r2mu_candidate_layers)})"
        )

    # --- Candidate protocol ---
    if not profile.candidate_positive:
        errors.append("candidate_positive is empty")
    if not profile.candidate_negative:
        errors.append("candidate_negative is empty")
    if profile.candidate_positive == profile.candidate_negative:
        errors.append(
            "candidate_positive and candidate_negative are identical"
        )

    return errors


def validate_environment_compatibility(
    profile: ModelFamilyProfile,
) -> list[str]:
    """Validate that the current runtime environment matches the profile.

    Checks ``transformers`` version against the profile's
    ``min_transformers_version`` and
    ``max_transformers_version_exclusive`` fields.

    Returns a list of error messages.  Empty list means the current
    environment is compatible with the profile.
    """
    errors: list[str] = []

    try:
        import transformers
        current_version = transformers.__version__
    except ImportError:
        errors.append("transformers is not installed")
        return errors

    from packaging.version import Version

    try:
        current = Version(current_version)
    except Exception:
        # Dev versions like '5.14.1.dev0' — strip .dev suffix
        base = current_version.split(".dev")[0]
        current = Version(base)

    if profile.min_transformers_version:
        min_ver = Version(profile.min_transformers_version)
        if current < min_ver:
            errors.append(
                f"transformers {current_version} < "
                f"min required {profile.min_transformers_version} "
                f"for {profile.key}"
            )

    if profile.max_transformers_version_exclusive:
        max_ver = Version(profile.max_transformers_version_exclusive)
        if current >= max_ver:
            errors.append(
                f"transformers {current_version} >= "
                f"max exclusive {profile.max_transformers_version_exclusive} "
                f"for {profile.key}"
            )

    return errors


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _validate_profile_types(
    lora: dict, structural: dict, access: dict, data: dict,
) -> None:
    """Validate YAML field types before dataclass construction."""
    for field_name, expected_type in [
        ("lora.rank", (int, float)),
        ("lora.alpha", (int, float)),
        ("lora.dropout", (int, float)),
        ("structural.r2mu_n_select_layers", (int, float)),
    ]:
        section, key = field_name.split(".", 1)
        section_dict = {"lora": lora, "structural": structural}[section]
        if key in section_dict:
            val = section_dict[key]
            if not isinstance(val, expected_type):
                raise ValueError(
                    f"{field_name} must be {expected_type}, "
                    f"got {type(val).__name__}: {val!r}"
                )

    for field_name in [
        "supports_prompting", "supports_candidate_margin",
        "supports_ga", "supports_gd", "supports_kl", "supports_npo",
        "supports_mmunlearner", "supports_manu", "supports_r2mu",
    ]:
        if field_name in data:
            val = data[field_name]
            if not isinstance(val, bool):
                raise ValueError(
                    f"{field_name} must be bool, "
                    f"got {type(val).__name__}: {val!r}"
                )

    if "requires_hf_auth" in access:
        val = access["requires_hf_auth"]
        if not isinstance(val, bool):
            raise ValueError(
                f"access.requires_hf_auth must be bool, "
                f"got {type(val).__name__}: {val!r}"
            )


def _safe_bool(val: Any) -> bool:
    """Coerce YAML value to bool, rejecting non-bool types."""
    if isinstance(val, bool):
        return val
    raise ValueError(f"Expected bool, got {type(val).__name__}: {val!r}")


def _safe_int(val: Any, field_name: str) -> int:
    """Coerce YAML value to int."""
    if isinstance(val, int):
        return val
    if isinstance(val, float) and val == int(val):
        return int(val)
    raise ValueError(
        f"{field_name} must be int, got {type(val).__name__}: {val!r}"
    )


def _safe_float(val: Any, field_name: str) -> float:
    """Coerce YAML value to float."""
    if isinstance(val, (int, float)):
        return float(val)
    raise ValueError(
        f"{field_name} must be numeric, got {type(val).__name__}: {val!r}"
    )


def _ensure_builtin_adapters_loaded() -> None:
    """Lazily import built-in adapter modules to trigger registration."""
    _try_import("qwen35")
    _try_import("glm46v")
    _try_import("internvl35")
    _try_import("phi4mm")
    _try_import("gemma3")


def _try_import(module_name: str) -> None:
    try:
        __import__(
            f"route_data.models.trainable.{module_name}",
            fromlist=[module_name],
        )
    except ImportError:
        pass


def clear_cache() -> None:
    """Clear the adapter cache. Useful for testing."""
    _ADAPTER_CACHE.clear()


# ------------------------------------------------------------------ #
# Runtime structural validation (P0-5)
# ------------------------------------------------------------------ #

def validate_structural_metadata(
    adapter: Any,
    model: Any,
) -> list[str]:
    """Validate profile structural metadata against the runtime model.

    Must be called after model load, before training begins.
    Returns a list of error messages.  Empty list means all metadata
    matches the runtime model.

    Checks
    ------
    - ``len(language_layers)`` matches ``profile.num_language_layers``
    - ``language_hidden_size`` matches ``profile.language_hidden_size``
    - ``language_intermediate_size`` matches ``profile.intermediate_size``
    - R²MU candidate indices are in range ``[0, num_language_layers)``
    """
    errors: list[str] = []
    p = adapter.profile

    # Only validate if profile has non-zero structural metadata
    if p.num_language_layers <= 0:
        return errors

    # Language layer count
    try:
        layers = adapter.language_layers(model)
        n_layers = len(layers)
    except (NotImplementedError, AttributeError) as exc:
        errors.append(f"Cannot validate layer count: {exc}")
        return errors

    if n_layers != p.num_language_layers:
        errors.append(
            f"num_language_layers mismatch: profile={p.num_language_layers}, "
            f"runtime={n_layers}"
        )

    # Hidden size
    if p.language_hidden_size > 0:
        try:
            runtime_hidden = adapter.language_hidden_size(model)
            if runtime_hidden != p.language_hidden_size:
                errors.append(
                    f"language_hidden_size mismatch: "
                    f"profile={p.language_hidden_size}, runtime={runtime_hidden}"
                )
        except (NotImplementedError, AttributeError) as exc:
            errors.append(f"Cannot validate hidden_size: {exc}")

    # Intermediate size
    if p.intermediate_size > 0:
        try:
            runtime_intermediate = adapter.language_intermediate_size(model)
            if runtime_intermediate != p.intermediate_size:
                errors.append(
                    f"intermediate_size mismatch: "
                    f"profile={p.intermediate_size}, runtime={runtime_intermediate}"
                )
        except (NotImplementedError, AttributeError) as exc:
            errors.append(f"Cannot validate intermediate_size: {exc}")

    # R²MU candidate index range validation
    if p.r2mu_candidate_layers:
        for idx in p.r2mu_candidate_layers:
            if idx < 0 or idx >= n_layers:
                errors.append(
                    f"r2mu_candidate_layers index {idx} out of range "
                    f"[0, {n_layers})"
                )
        if len(p.r2mu_candidate_layers) != len(set(p.r2mu_candidate_layers)):
            errors.append("r2mu_candidate_layers contains duplicates")

    # Capability-scoped: R²MU
    if p.supports_r2mu:
        if not p.r2mu_candidate_layers:
            errors.append(
                "supports_r2mu=true but r2mu_candidate_layers is empty"
            )
        elif (
            isinstance(p.r2mu_n_select_layers, int)
            and isinstance(p.r2mu_candidate_layers, (list, tuple))
            and p.r2mu_n_select_layers > len(p.r2mu_candidate_layers)
        ):
            errors.append(
                f"r2mu_n_select_layers ({p.r2mu_n_select_layers}) > "
                f"number of candidates ({len(p.r2mu_candidate_layers)})"
            )

    # Capability-scoped: MANU
    if p.supports_manu and isinstance(p.intermediate_size, int) and p.intermediate_size <= 0:
        errors.append(
            "supports_manu=true but intermediate_size is not resolved"
        )

    return errors
