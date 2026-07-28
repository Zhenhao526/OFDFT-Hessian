from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "qm9_complete_total_stage2_provenance_audit.py"
    spec = spec_from_file_location("stage2_provenance", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_allows_value_exact_float_precision_promotion():
    module = _load_script()
    source = {
        "weight": torch.tensor([1.25, -2.5], dtype=torch.float32),
        "count": torch.tensor([3], dtype=torch.int64),
    }
    promoted = {
        "weight": source["weight"].to(torch.float64),
        "count": source["count"].clone(),
    }
    result = module._compare_state_dicts(source, promoted)
    assert result["exact_match"] is True
    assert result["dtype_difference_count"] == 1
    assert module._canonical_state_dict_sha256(source) == module._canonical_state_dict_sha256(
        promoted
    )


def test_compare_rejects_parameter_value_change():
    module = _load_script()
    source = {"weight": torch.tensor([1.0], dtype=torch.float32)}
    changed = {"weight": torch.tensor([1.001], dtype=torch.float64)}
    result = module._compare_state_dicts(source, changed)
    assert result["exact_match"] is False
    assert result["mismatched_keys"] == ["weight"]
    assert result["max_abs_parameter_difference"] > 0.0
