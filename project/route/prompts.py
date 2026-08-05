"""Training-condition prompts and answers (PLAN.md section 6).

Conditions:
    direct   (C1): image -> property            ("What property is shown?")
    joint    (C2): image -> identity -> property in one assistant sequence
    mediated (C3): disjoint image->identity (C3a) + text alias->property (C3b)
    mixed    (C4): direct + image->identity + alias->property examples

Key invariant for mediated/mixed: no example contains both an image and its
target property in the same training context.
"""

from __future__ import annotations

PROPERTY_QUESTION = "What property is shown?"
IDENTITY_QUESTION = "Who is this?"
JOINT_QUESTION = "Identify this person and state their property."
ALIAS_PROPERTY_QUESTION = "What property does {alias} have?"

# Example types produced by build_examples()
DIRECT = "direct"                 # image -> property
IDENTITY = "identity"             # image -> alias          (C3a)
ALIAS_PROPERTY = "alias_property" # alias -> property       (C3b, text-only)
JOINT = "joint"                   # image -> alias + property sequence

# Variants used per example type during training (PLAN.md sections 4, 6)
TRAIN_VARIANT = {
    DIRECT: "aligned",        # marker consistent with identity property
    JOINT: "aligned",
    IDENTITY: "random_marker",  # marker independent of identity (or absent)
    ALIAS_PROPERTY: None,       # text-only, no image
}


def direct_example(alias: str, prop: str) -> tuple[str, str]:
    return PROPERTY_QUESTION, prop


def identity_example(alias: str) -> tuple[str, str]:
    return IDENTITY_QUESTION, alias


def alias_property_example(alias: str, prop: str) -> tuple[str, str]:
    return ALIAS_PROPERTY_QUESTION.format(alias=alias), prop


def joint_example(alias: str, prop: str) -> tuple[str, str]:
    answer = f"This is {alias}. {alias} has {prop}."
    return JOINT_QUESTION, answer


def build_examples(condition: str, rows: list[dict]) -> list[dict]:
    """Expand manifest rows into pre-tokenization training examples.

    Returns a list of dicts:
        {row, question, answer, example_type, variant}
    `variant` is None for text-only examples.
    """
    examples = []

    def add(row, example_type):
        question, answer = _make_example(example_type, row["alias"], row["property"])
        examples.append({
            "row": row,
            "question": question,
            "answer": answer,
            "example_type": example_type,
            "variant": TRAIN_VARIANT[example_type],
        })

    for row in rows:
        if condition == "direct":
            add(row, DIRECT)
        elif condition == "joint":
            add(row, JOINT)
        elif condition == "mediated":
            add(row, IDENTITY)
            add(row, ALIAS_PROPERTY)
        elif condition == "mixed":
            add(row, DIRECT)
            add(row, IDENTITY)
            add(row, ALIAS_PROPERTY)
        else:
            raise ValueError(f"Unknown condition: {condition!r}")

    return examples


def _make_example(example_type: str, alias: str, prop: str) -> tuple[str, str]:
    if example_type == DIRECT:
        return direct_example(alias, prop)
    if example_type == IDENTITY:
        return identity_example(alias)
    if example_type == ALIAS_PROPERTY:
        return alias_property_example(alias, prop)
    if example_type == JOINT:
        return joint_example(alias, prop)
    raise ValueError(f"Unknown example_type: {example_type!r}")


# ---------------------------------------------------------------------------
# Evaluation prompts (PLAN.md section 13)
# ---------------------------------------------------------------------------

EVAL_VARIANTS = [
    "aligned",
    "conflict",
    "no_marker",
    "face_masked",
    "face_masked_no_marker",
    "neutral_marker",
]


def eval_property_question() -> str:
    return PROPERTY_QUESTION


def eval_identity_question() -> str:
    return IDENTITY_QUESTION
