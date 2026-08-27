"""Stage 6 tests: leave-one-out stability."""

import pandas as pd

from rlens.small_sample import (leave_one_prompt_out, leave_one_set_out,
                                prompt_level_effects, sign_test, summarise)


def _panel(deltas=None):
    """Two models x 5 sets x 4 prompts x 5 depths, R above J by `deltas`."""
    rows = []
    for model in ("qwen", "gemma"):
        for set_name in ("association", "multihop", "multilingual", "poetry", "typo"):
            for item in range(4):
                gap = (deltas or {}).get((set_name, str(item)), 1.0)
                for depth in (0.0, 0.1, 0.2, 0.3, 0.4):
                    for lens, base in (("logit", 0.0), ("released-J", 1.0),
                                       ("released-R", 1.0 + gap)):
                        rows.append({
                            "cell_id": f"{model}-{set_name}-{item}-{depth}",
                            "model_key": model, "set": set_name, "item_id": str(item),
                            "requested_depth": depth, "layer": int(depth * 10),
                            "lens": lens, "contextual_coherence": base})
    return pd.DataFrame(rows)


def test_leave_one_prompt_out_covers_every_prompt():
    out = leave_one_prompt_out(_panel(), "released-R", "released-J",
                               "contextual_coherence")
    assert len(out) == 20
    assert out["n_prompts"].eq(19).all(), "exactly one prompt removed each time"


def test_deleting_a_prompt_removes_all_of_its_depths():
    """The prompt is the cluster. Removing only some of its rows would leave a
    partially-deleted cluster and understate its influence."""
    panel = _panel()
    out = leave_one_prompt_out(panel[panel["model_key"] == "qwen"],
                               "released-R", "released-J", "contextual_coherence")
    # 20 prompts x 5 depths = 100 cells; dropping one prompt leaves 95
    assert out["n_cells"].eq(95).all()


def test_an_influential_prompt_is_identified():
    """One prompt carrying the whole effect must show up as the deletion that
    drops the estimate furthest."""
    panel = _panel({("poetry", "0"): 20.0})
    out = leave_one_prompt_out(panel[panel["model_key"] == "qwen"],
                               "released-R", "released-J", "contextual_coherence")
    assert summarise(out)["most_influential"] == "poetry::0"


def test_leave_one_set_out_reweights_the_remainder_equally():
    out = leave_one_set_out(_panel(), "released-R", "released-J",
                            "contextual_coherence")
    assert len(out) == 5 and out["n_sets"].eq(4).all()


def test_summarise_flags_when_every_deletion_stays_positive():
    out = leave_one_prompt_out(_panel(), "released-R", "released-J",
                               "contextual_coherence")
    assert summarise(out)["all_positive"] is True


def test_summarise_flags_a_sign_change():
    panel = _panel({("typo", "0"): -30.0})
    out = leave_one_prompt_out(panel[panel["model_key"] == "qwen"],
                               "released-R", "released-J", "contextual_coherence")
    assert summarise(out)["all_positive"] is False


def test_prompt_level_effects_average_depths_not_count_them():
    """Five depths are repeated measurements of one prompt, not five
    observations."""
    out = prompt_level_effects(_panel(), "released-R", "released-J",
                               "contextual_coherence")
    assert len(out) == 40                      # 2 models x 20 prompts
    assert out["n_depths"].eq(5).all()


def test_sign_test_is_exact_and_reports_ties():
    assert sign_test([1, 1, 1, 1, -1])["n_positive"] == 4
    out = sign_test([1, -1, 0, 0])
    assert out["n_tied"] == 2 and out["n_nonzero"] == 2
    assert sign_test([])["p_value"] is None


def test_sign_test_p_value_matches_the_exact_binomial():
    out = sign_test([1] * 10)
    assert out["p_value"] == 2 * (0.5 ** 10)
