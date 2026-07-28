from __future__ import annotations

import json

from scripts.qm9_complete_total_replay_transient_rescue import _retry_reason


def test_retry_reason_only_selects_missing_or_transient_failures(tmp_path):
    missing = tmp_path / "missing.json"
    assert _retry_reason(missing) == "missing_summary"

    success = tmp_path / "success.json"
    success.write_text(json.dumps({"success": True}))
    assert _retry_reason(success) is None

    transient = tmp_path / "transient.json"
    transient.write_text(json.dumps({"success": False, "error_type": "OSError"}))
    assert _retry_reason(transient) == "transient_OSError"

    structural = tmp_path / "structural.json"
    structural.write_text(json.dumps({"success": False, "strict_converged": False}))
    assert _retry_reason(structural) is None
