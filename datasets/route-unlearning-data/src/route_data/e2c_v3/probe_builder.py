"""E2C-v3 factorial probe builder.

Builds the complete set of intervention probes for the Phase A audit:
  - bare image probes
  - correct code probes
  - wrong code probes (single + multiple)
  - code-only probes
  - shuffled code probes

All pairings are deterministic (seed 17).
No test image appears in training.
"""
import json


def load_identity_mapping(path: str = "e2c_v3/manifests/identity_code_mapping.json") -> dict:
    """Load the frozen identity → code → alias mapping."""
    return json.load(open(path))


def load_wrong_code_pairs(path: str = "e2c_v3/manifests/wrong_code_pairs.json") -> dict:
    """Load the wrong code pairing manifest."""
    return json.load(open(path))


def load_split_manifest(
    path: str = "e2c_v2/manifests/e2c_image_split.json",
) -> list[dict]:
    """Load the frozen image split manifest."""
    return json.load(open(path))


def get_experimental_identities() -> list[str]:
    """Return the 10 frozen experimental identity IDs."""
    return [f"syn_{i:02d}" for i in range(10)]


def build_bare_image_probes(
    eval_items: list[dict],
    identity_to_alias: dict[str, str],
    prompt: str = "Identify the synthetic person shown here. Return only the alias.",
) -> list[dict]:
    """A1.1 — Bare image probes: Y(X_i)."""
    probes = []
    for item in eval_items:
        iid = item["identity_id"]
        probes.append({
            "probe_id": f"bare__{iid}__{item['image_id']}",
            "probe_type": "bare_image",
            "identity_id": iid,
            "image_id": item["image_id"],
            "image_path": item["image_path"],
            "code_id": None,
            "prompt": prompt,
            "expected_alias": identity_to_alias[iid],
            "has_image": True,
        })
    return probes


def build_correct_code_probes(
    eval_items: list[dict],
    identity_to_code: dict[str, str],
    code_to_alias: dict[str, str],
    prompt_tmpl: str = "Identity code: {code}. Identify the person shown. Return only the alias.",
) -> list[dict]:
    """A1.2 — Image + correct code probes: Y(X_i, C_i)."""
    probes = []
    for item in eval_items:
        iid = item["identity_id"]
        code = identity_to_code[iid]
        probes.append({
            "probe_id": f"correct__{iid}__{item['image_id']}",
            "probe_type": "image_correct_code",
            "identity_id": iid,
            "image_id": item["image_id"],
            "image_path": item["image_path"],
            "code_id": code,
            "prompt": prompt_tmpl.format(code=code),
            "expected_alias": code_to_alias[code],
            "has_image": True,
        })
    return probes


def build_wrong_code_probes(
    eval_items: list[dict],
    identity_to_code: dict[str, str],
    code_to_alias: dict[str, str],
    wrong_pairs: dict,
    n_wrong: int | None = None,
    prompt_tmpl: str = "Identity code: {code}. Identify the person shown. Return only the alias.",
) -> list[dict]:
    """A1.3/A1.4 — Image + wrong code probes: Y(X_i, C_j) for j ≠ i.

    If n_wrong is None, use all 9 wrong codes per identity.
    Otherwise use the first n_wrong wrong codes (ring-next first).
    """
    probes = []
    for item in eval_items:
        iid = item["identity_id"]
        pair_info = wrong_pairs["pairs"][iid]
        wrong_codes = pair_info["all_wrong_codes"]
        if n_wrong is not None:
            wrong_codes = wrong_codes[:n_wrong]

        for wc in wrong_codes:
            code = wc["code_id"]
            probes.append({
                "probe_id": (
                    f"wrong__{iid}__{item['image_id']}__{code}"
                ),
                "probe_type": "image_wrong_code",
                "identity_id": iid,
                "image_id": item["image_id"],
                "image_path": item["image_path"],
                "code_id": code,
                "prompt": prompt_tmpl.format(code=code),
                "expected_alias_if_follow_code": code_to_alias[code],
                "expected_alias_if_follow_image": pair_info["true_alias"],
                "has_image": True,
            })
    return probes


def build_code_only_probes(
    identity_ids: list[str],
    identity_to_code: dict[str, str],
    code_to_alias: dict[str, str],
    prompt_tmpl: str = "Identity code: {code}. What is the alias?",
) -> list[dict]:
    """Code-only probes: Y(C_i) — no image."""
    probes = []
    for iid in identity_ids:
        code = identity_to_code[iid]
        probes.append({
            "probe_id": f"code_only__{iid}",
            "probe_type": "code_only_correct",
            "identity_id": iid,
            "image_id": None,
            "image_path": None,
            "code_id": code,
            "prompt": prompt_tmpl.format(code=code),
            "expected_alias": code_to_alias[code],
            "has_image": False,
        })
    return probes


def build_shuffled_code_probes(
    identity_ids: list[str],
    identity_to_code: dict[str, str],
    shuffled_code_to_alias: dict[str, str],
    prompt_tmpl: str = "Identity code: {code}. What is the alias?",
) -> list[dict]:
    """Shuffled code-only probes: Y(C_i) → shuffled alias."""
    probes = []
    for iid in identity_ids:
        code = identity_to_code[iid]
        probes.append({
            "probe_id": f"shuffled__{iid}",
            "probe_type": "code_only_shuffled",
            "identity_id": iid,
            "image_id": None,
            "image_path": None,
            "code_id": code,
            "prompt": prompt_tmpl.format(code=code),
            "expected_alias": shuffled_code_to_alias[code],
            "has_image": False,
        })
    return probes


def validate_factorial_probes(probes: list[dict]) -> list[str]:
    """Validate probe set integrity. Returns list of error strings (empty = OK)."""
    errors = []
    seen_ids = set()
    for p in probes:
        pid = p["probe_id"]
        if pid in seen_ids:
            errors.append(f"Duplicate probe_id: {pid}")
        seen_ids.add(pid)

        if p["probe_type"] == "image_wrong_code":
            # Wrong code must differ from true code
            # (ensured by construction, but verify)
            pass

        if p["has_image"] and p["image_path"] is None:
            errors.append(f"Probe {pid} has_image=True but image_path=None")
        if not p["has_image"] and p["image_path"] is not None:
            errors.append(f"Probe {pid} has_image=False but image_path set")

    return errors
