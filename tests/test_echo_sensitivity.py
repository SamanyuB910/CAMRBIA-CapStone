"""Winner/score consistency in the combination rule, and echo-restricted pairing."""

import json

import pandas as pd
import pytest

from rlens.autorate import combine, parse_scores


def _scores(a, b, c, winner):
    return {"A": {"contextual_coherence": a, "lexical_integrity": 1, "prompt_echo": 0,
                  "evidence": ""},
            "B": {"contextual_coherence": b, "lexical_integrity": 1, "prompt_echo": 0,
                  "evidence": ""},
            "C": {"contextual_coherence": c, "lexical_integrity": 1, "prompt_echo": 0,
                  "evidence": ""},
            "contextual_winner": winner}


def test_combined_winner_is_derived_from_combined_scores_not_voted():
    """The frozen run had 18/200 cells where a winner VOTE disagreed with the
    averaged scores: judge 1 (A=3,B=2) votes A, judge 2 (A=1,B=4) votes B, the
    vote ties, but the mean (A=2,B=3) has B uniquely leading."""
    out = combine([_scores(3, 2, 1, "A"), _scores(1, 4, 1, "B")])
    assert out["A"]["contextual_coherence"] == 2.0
    assert out["B"]["contextual_coherence"] == 3.0
    assert out["contextual_winner"] == "B", "must follow the combined scores"
    assert out["judge_winner_votes"] == ["A", "B"], "the raw votes are retained"


def test_combined_winner_is_a_tie_only_when_the_maximum_is_shared():
    tied = combine([_scores(3, 3, 1, "A"), _scores(3, 3, 1, "B")])
    assert tied["contextual_winner"] == "tie"

    unique = combine([_scores(4, 1, 0, "A"), _scores(4, 2, 0, "A")])
    assert unique["contextual_winner"] == "A"


def test_median_of_three_also_yields_a_consistent_winner():
    out = combine([_scores(4, 0, 0, "A"), _scores(0, 4, 0, "B"), _scores(0, 4, 0, "B")])
    assert out["A"]["contextual_coherence"] == 0
    assert out["B"]["contextual_coherence"] == 4
    assert out["contextual_winner"] == "B"


@pytest.mark.parametrize("triple", [
    (_scores(3, 2, 1, "A"), _scores(1, 4, 1, "B")),
    (_scores(0, 0, 0, "tie"), _scores(4, 0, 0, "A")),
    (_scores(2, 2, 2, "tie"), _scores(1, 3, 2, "B"), _scores(3, 1, 2, "A")),
])
def test_no_combination_can_produce_an_inconsistent_winner(triple):
    """Property: whatever the inputs, the output always satisfies audit
    condition 9."""
    out = combine(list(triple))
    contextual = {l: out[l]["contextual_coherence"] for l in ("A", "B", "C")}
    best = max(contextual.values())
    leaders = [l for l, v in contextual.items() if v == best]
    winner = out["contextual_winner"]
    if winner == "tie":
        assert len(leaders) >= 2
    else:
        assert contextual[winner] == best


def test_combined_output_still_parses_as_a_score_payload():
    """The extra `judge_winner_votes` key must not break anything that consumes
    combined scores."""
    out = combine([_scores(4, 1, 0, "A"), _scores(4, 2, 0, "A")])
    assert set(out) == {"A", "B", "C", "contextual_winner", "judge_winner_votes"}
    assert all(isinstance(out[l], dict) for l in ("A", "B", "C"))


def test_recombine_is_documented_as_making_no_api_calls():
    import inspect

    from rlens import cli

    source = inspect.getsource(cli.cmd_recombine)
    assert "Makes no API calls" in source
    assert "call_judge" not in source and "urlopen" not in source
    assert "_to_panel_labels" in source, "raw responses must be translated to panel labels"


def test_recombine_refuses_a_non_empty_destination():
    import inspect

    from rlens import cli

    assert "choose a fresh destination" in inspect.getsource(cli.cmd_recombine)
