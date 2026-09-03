"""Unit tests for the CPU-only matrix headline verification/repair helper.

The key repaired semantics: h-side output-health counts (unparseable,
multi-label) cover ONLY rows where h was actually queried (g routed);
unrouted rows are g failures and must not inflate h counts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_V_PATH = (Path(__file__).resolve().parents[2]
           / "scripts" / "verify_matrix_headlines.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_mx_under_test",
                                                  _V_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vm = _load_module()


def test_recount_e2e_excludes_unrouted_rows():
    rows = [
        # routed, parsed fine
        {"g_routed": True, "pred_code": "PPU_001", "pred_alias": "Oden",
         "multi_label_ambiguous": False},
        # routed, h output unparseable
        {"g_routed": True, "pred_code": "PPU_002", "pred_alias": None,
         "multi_label_ambiguous": False},
        # routed, ambiguous multi-label output
        {"g_routed": True, "pred_code": "PPU_003", "pred_alias": None,
         "multi_label_ambiguous": True},
        # NOT routed: h never queried (pred_alias None by construction)
        {"g_routed": False, "pred_code": None, "pred_alias": None},
        # routed flag true but no code (hallucinated/None code path)
        {"g_routed": True, "pred_code": None, "pred_alias": None},
    ]
    rc = vm.recount_e2e(rows)
    assert rc["h_unparseable_outputs"] == 2      # only the two routed rows
    assert rc["h_multi_label_ambiguous_outputs"] == 1
    assert rc["g_routing_failures"] == 2         # unrouted + codeless
    assert rc["n_rows"] == 5


def test_recount_e2e_all_routed_clean():
    rows = [{"g_routed": True, "pred_code": f"C{i}", "pred_alias": "A",
             "multi_label_ambiguous": False} for i in range(4)]
    rc = vm.recount_e2e(rows)
    assert rc == {"h_unparseable_outputs": 0,
                  "h_multi_label_ambiguous_outputs": 0,
                  "g_routing_failures": 0, "n_rows": 4}
