"""The strict parser is reachable without importing torch.

The parser decided that a verbatim comma-bearing SOC title was unparseable and
cost 12 of 30 MLLMU target rows, so it is the one piece of scoring code whose
behavior has already produced a wrong verdict once.  It also sits on the
cheapest path in the project: ``--verify`` re-derives the frozen pilot design
on CPU, the design tests are documented as running "without a GPU, without
torch", and GX2P re-parses stored cells from the raw text they recorded.  All
three used to reach the parser through ``e2c_v3_research_validity``, which
imports torch at module scope -- so each of them paid the training stack's
import cost to run three string functions, and "lightweight" was a claim about
intent rather than about what executed.

These tests prove the extraction by running the modules in a SUBPROCESS with
``sys.modules["torch"] = None``, which makes any ``import torch`` raise
ImportError.  An in-process assertion could not prove it: pytest has already
imported torch by the time this file runs, so the module would load from
``sys.modules`` and the dependency would be invisible.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
PARSER = _SCRIPTS / "e2c_v3_label_parser.py"

#: The broad SOC title that is the correct post-edit answer for two of the five
#: MLLMU pilot sets, and that the asymmetric matcher scored as unparseable.
COMMA_LABEL = "Software and Web Developers, Programmers, and Testers"
VOCAB = ["Software Developer", COMMA_LABEL, "Unknown"]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_without_torch(body):
    """Exec ``body`` in a subprocess where importing torch is impossible.

    Returns (stdout, CompletedProcess).  ``sys.modules[name] = None`` is the
    documented way to make ``import name`` raise ImportError, and it also
    defeats ``from torch.utils.data import ...``.
    """
    preamble = (
        "import sys\n"
        "for _m in ('torch', 'torch.nn', 'torch.utils', 'torch.utils.data',\n"
        "           'transformers', 'numpy'):\n"
        "    sys.modules[_m] = None\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", preamble + body],
        capture_output=True, text=True, cwd=str(_ROOT), timeout=300,
        check=False)
    return proc.stdout.strip(), proc


def test_the_parser_module_itself_needs_no_torch():
    body = (
        "import importlib.util, json\n"
        f"spec = importlib.util.spec_from_file_location('lp', {str(PARSER)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        f"print(json.dumps({{'parsed': m.parse_recognized_label({COMMA_LABEL!r},"
        f" {VOCAB!r}),\n"
        f"                          'unparseable':"
        f" m.check_vocab_parseable({VOCAB!r}),\n"
        f"                          'recognized': m.recognized_labels_in("
        f"'the {COMMA_LABEL}', {VOCAB!r})}}))\n"
    )
    out, proc = _run_without_torch(body)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(out)
    # the comma-bearing label parses, and the vocab invariant is clean: the
    # extraction must not regress the fix it was made to protect
    assert got["parsed"] == COMMA_LABEL
    assert got["unparseable"] == []
    assert got["recognized"] == [COMMA_LABEL]


def test_the_design_freeze_module_needs_no_torch():
    """``e2c_v3_mllmu_matrix --verify`` re-derives the frozen pilot on CPU.
    Its only torch dependency was the parser, reached through the scoring
    script; loading the whole module with torch blocked is what makes
    "genuinely lightweight" a tested property instead of a docstring."""
    g6m_path = repr(str(_SCRIPTS / "e2c_v3_mllmu_matrix.py"))
    vocab = repr(VOCAB)
    body = (
        "import importlib.util, json\n"
        "spec = importlib.util.spec_from_file_location(\n"
        "    'g6m', " + g6m_path + ")\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "print(json.dumps({\n"
        "    'parser_is_the_torch_free_module':\n"
        "        m.parser_module().__name__ == 'e2c_v3_label_parser',\n"
        "    'comma_label_parseable':\n"
        "        m.parser_module().check_vocab_parseable(" + vocab + ") == [],\n"
        # the sentinel is still None: nothing on the import path replaced it
        # with a real torch module, and no submodule resolved either
        "    'torch_still_blocked': sys.modules.get('torch') is None,\n"
        "    'no_torch_submodule': not any(\n"
        "        k.startswith('torch.') and sys.modules[k] is not None\n"
        "        for k in sys.modules)}))\n"
    )
    out, proc = _run_without_torch(body)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(out)
    assert got["parser_is_the_torch_free_module"] is True
    assert got["comma_label_parseable"] is True
    # nothing on the path pulled torch in as a side effect
    assert got["torch_still_blocked"] is True
    assert got["no_torch_submodule"] is True


def test_the_scoring_script_reexports_the_same_function_objects():
    """Two copies of the normalization rule is how the matcher's two sides
    drifted apart in the first place, so the re-export must be the same object,
    not a re-implementation or a wrapper that could diverge."""
    rv = _load("rv_reexport_under_test", "e2c_v3_research_validity.py")
    lp = rv._label_parser          # the module rv actually resolved
    assert lp.__name__ == "e2c_v3_label_parser"
    assert rv.recognized_labels_in is lp.recognized_labels_in
    assert rv.parse_recognized_label is lp.parse_recognized_label
    assert rv.check_vocab_parseable is lp.check_vocab_parseable
    # a second, independently loaded copy of the same bytes must agree, so the
    # re-export cannot be a different rule that happens to share a name
    fresh = _load("lp_fresh_under_test", "e2c_v3_label_parser.py")
    assert fresh.parse_recognized_label(COMMA_LABEL, VOCAB) == \
        rv.parse_recognized_label(COMMA_LABEL, VOCAB) == COMMA_LABEL
    # and the digest it pins is the digest of the module that holds the bytes
    assert rv.label_parser_sha256() == _sha256(PARSER)
    assert rv.label_parser_sha256() != rv.script_sha256()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_the_parser_module_source_imports_nothing_heavy():
    """A guard that fails at review time rather than at the next GPU run: the
    module's whole purpose is to be importable where torch is not.

    Checked by parsing the AST rather than by searching the text, because the
    module's own docstring explains which imports are forbidden -- a substring
    search would trip over the explanation.
    """
    tree = ast.parse(PARSER.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported == set(), sorted(imported)
    # no I/O either: the parser is a pure function of (text, vocab)
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "open" not in calls


def test_the_runner_declares_and_pins_the_parser_module():
    """The parser decides what counted as a recognized label, so a run that
    did not pin its bytes could not be reproduced from its own provenance."""
    gxm = _load("gxm_parser_pin_under_test",
                "e2c_v3_granularity_matrix.py")
    assert "scripts/e2c_v3_label_parser.py" in gxm.GX_CODE
    rv = gxm.rv
    assert rv.label_parser_sha256() == _sha256(PARSER)
