"""Config echo: our serialized rule config must match the released
``provenance.config_json`` for the dense models, field for field.

Requires results/provenance_qwen3.5-4b.json (produced by scripts/smoke_test.py).
"""

import json
from pathlib import Path

import pytest

from rlens.rules import RulesConfig

PROVENANCE = Path(__file__).resolve().parents[1] / "results" / "provenance_qwen3.5-4b.json"


@pytest.fixture(scope="module")
def released():
    if not PROVENANCE.exists():
        pytest.skip("run scripts/smoke_test.py first to dump released provenance")
    return json.loads(PROVENANCE.read_text(encoding="utf-8"))


def test_j_lens_config_matches_released(released):
    theirs = released["j-lens"]["config_json"]
    assert RulesConfig.all_off().to_config_json() == theirs
    assert json.loads(theirs) == {"estimator": "standard"}


def test_r_lens_config_matches_released(released):
    theirs = released["r-lens"]["config_json"]
    ours = RulesConfig().to_config_json()
    assert json.loads(ours) == json.loads(theirs), "field/value mismatch vs released artifact"
    assert ours == theirs, "byte-level mismatch (key order or float formatting)"


def test_roundtrip_from_released(released):
    theirs = released["r-lens"]["config_json"]
    cfg = RulesConfig.from_config_json(theirs)
    assert cfg == RulesConfig()  # released R-lens == our defaults
    assert cfg.to_config_json() == theirs

    assert RulesConfig.from_config_json('{"estimator": "standard"}') == RulesConfig.all_off()


def test_released_recipe_fields(released):
    """Pin the released fitting recipe our fit wrapper must reproduce."""
    for arm in ("j-lens", "r-lens"):
        prov = released[arm]
        assert prov["model_id"] == "Qwen/Qwen3.5-4B"
        assert prov["dataset_id"] == "NeelNanda/pile-10k"
        assert prov["target_layer"] == 30
        assert prov["skip_first"] == 4
        assert prov["t_max"] == 128
        assert prov["n_prompts"] == 25
        # docs_consumed == n_prompts: prompts were taken sequentially from the
        # start of the corpus with none skipped -> rows [0:25].
        assert prov["docs_consumed"] == 25
        assert prov["weighting"] == "uniform"
        assert prov["corpus_mode"] == "pretrain"
