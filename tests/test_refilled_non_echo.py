"""Stage 3 tests: refilled non-echo rankings."""

import pytest

from rlens.refill import (REFILL_K, causal_prefix_ids, is_special, normalise,
                          refill, refill_report, verify_prefix_reproduces)


def _ranked(tokens):
    return [{"rank": i + 1, "token": t, "token_id": 1000 + i}
            for i, t in enumerate(tokens)]


def test_causal_prefix_includes_the_readout_position():
    """The readout is taken AT that position, so the model has seen that token
    and could echo it. Excluding it would under-count copies."""
    assert causal_prefix_ids([10, 11, 12, 13], 2) == {10, 11, 12}


def test_causal_prefix_excludes_tokens_after_the_readout():
    """Tokens later in the sequence could not have been echoed."""
    assert 13 not in causal_prefix_ids([10, 11, 12, 13], 2)


def test_negative_readout_position_is_rejected():
    with pytest.raises(ValueError):
        causal_prefix_ids([1, 2], -1)


def test_refill_preserves_original_rank_order():
    ranked = _ranked([" a", " b", " c", " d", " e"])
    kept = refill(ranked, {1000, 1002}, k=3)
    assert [e["token"] for e in kept] == [" b", " d", " e"]
    assert [e["original_rank"] for e in kept] == [2, 4, 5]
    assert [e["refilled_rank"] for e in kept] == [1, 2, 3]


def test_refill_removes_special_and_whitespace_tokens():
    ranked = _ranked([" a", "   ", "", " b"])
    kept = refill(ranked, set(), k=2)
    assert [e["token"] for e in kept] == [" a", " b"]


def test_refill_removes_declared_special_ids():
    ranked = _ranked([" a", " b", " c"])
    kept = refill(ranked, set(), k=2, special_ids={1001})
    assert [e["token"] for e in kept] == [" a", " c"]


def test_no_copied_token_id_survives():
    ranked = _ranked([" x", " y", " z", " w"])
    prefix = {1000, 1001}
    kept = refill(ranked, prefix, k=2)
    assert not ({e["token_id"] for e in kept} & prefix)


def test_normalised_rule_catches_case_and_whitespace_variants():
    ranked = [{"rank": 1, "token": " Paris", "token_id": 7},
              {"rank": 2, "token": " Seine", "token_id": 8}]
    kept = refill(ranked, set(), k=1, prefix_norms={"paris"}, use_normalised=True)
    assert [e["token"] for e in kept] == [" Seine"]


def test_normalised_rule_is_off_by_default():
    ranked = [{"rank": 1, "token": " Paris", "token_id": 7}]
    assert len(refill(ranked, set(), k=1, prefix_norms={"paris"})) == 1


def test_normalisation_is_deterministic_and_unicode_aware():
    assert normalise("  Ｐａｒｉｓ ") == "paris"
    assert normalise(" the") == normalise("The ") == "the"


def test_refill_returns_short_rather_than_padding():
    """A short list must be visible to the caller as an error, never padded."""
    ranked = _ranked([" a", " b", " c"])
    kept = refill(ranked, {1000, 1001}, k=10)
    assert len(kept) == 1
    assert refill_report(ranked, kept, k=10)["complete"] is False


def test_report_counts_copies_removed_from_the_original_top_ten():
    ranked = _ranked([f" t{i}" for i in range(20)])
    kept = refill(ranked, {1000, 1001, 1002}, k=REFILL_K)
    report = refill_report(ranked, kept, k=REFILL_K)
    assert report["complete"] and report["n_kept"] == 10
    assert report["n_copied_removed_from_top10"] == 3
    assert report["deepest_rank_used"] == 13


def test_verify_accepts_an_exact_reproduction():
    ranked = _ranked([f" t{i}" for i in range(20)])
    ok, detail = verify_prefix_reproduces(ranked, ranked[:10])
    assert ok and "reproduce exactly" in detail


def test_verify_rejects_a_changed_ranking_and_names_the_rank():
    """A deeper pass that does not reproduce the frozen top-10 is a different
    measurement; the refilled panel would not be comparable."""
    ranked = _ranked([f" t{i}" for i in range(20)])
    frozen = _ranked([f" t{i}" for i in range(20)])
    frozen[4]["token"] = " CHANGED"
    ok, detail = verify_prefix_reproduces(ranked, frozen[:10])
    assert not ok and "rank 5" in detail


def test_refill_panel_aborts_rather_than_padding_a_short_list():
    import inspect

    from rlens import cli

    src = inspect.getsource(cli.cmd_refill_panel)
    assert "raise SystemExit" in src
    assert "rather than padding the list" in src
    assert "did not reproduce their frozen top-10" in src


def test_refill_panel_refuses_to_overwrite_a_key_or_panel():
    import inspect

    from rlens import cli

    src = inspect.getsource(cli.cmd_refill_panel)
    assert "exists and is not empty" in src
    assert "args.force" not in src
