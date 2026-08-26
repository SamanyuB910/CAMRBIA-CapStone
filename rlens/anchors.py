"""Harness validation against the published qualitative anchors.

Before any null result from this repo can be believed, the pipeline has to
reproduce claims the source papers state exactly. The R-lens post makes five
of them, each naming a prompt, a *token position inside that prompt*, a concept
token, and the layer at which each lens surfaces it. They are quoted verbatim
below and checked mechanically.

Two things make this the right validation instrument:

* It reads at the position the post actually names ("on the token 'sushi'"),
  not at the answer position the pass@10 protocol uses. The post's qualitative
  claims live in the layer x position grid, and this is the only place in the
  repo that samples it.
* It is directional. The post's examples are on its headline model
  (Qwen3.6-27B) or an unspecified one, so exact layer numbers need not transfer
  to another 27B. What must transfer is the *ordering*: R-lens surfacing the
  concept substantially earlier than J-lens. A run that reproduces the ordering
  validates the harness; one that inverts it indicates a pipeline fault.

Anchors whose prompt the post does not print are marked ``reconstructed`` and
must be reported separately -- their wording is ours, so a miss is weak
evidence of anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Anchor:
    name: str
    prompt: str
    position_token: str   # the token the post says it reads at
    concept: str          # the concept the lens should surface
    quote: str            # verbatim from the source
    reported: dict = field(default_factory=dict)  # lens -> reported layer (None = "never")
    criterion: str = "top10"                      # "top1" or "top10"
    reconstructed: bool = False                   # True if we wrote the prompt


ANCHORS: tuple[Anchor, ...] = (
    Anchor(
        name="multihop-sushi",
        prompt="The capital of the country where sushi originated is",
        position_token=" sushi",
        concept=" Japan",
        criterion="top10",
        reported={"R": 2, "J": 14},
        quote=(
            "On the token “sushi”, R-lens surfaces the intermediate “Japan” at "
            "layer 2, while J-lens does not surface it until layer 14"
        ),
    ),
    Anchor(
        name="assoc-jordan",
        prompt="The athlete Michael Jordan plays the sport of",
        position_token=" Jordan",
        concept=" basketball",
        criterion="top10",
        reported={"R": 4, "J": 20},
        quote=(
            "On the token “Jordan”, R-lens surfaces the token “basketball” at "
            "layer 4, while J-lens surfaces it at layer 20"
        ),
    ),
    Anchor(
        name="typo-aganst",
        prompt="I have nothing aganst him personally, but",
        position_token=" aganst",
        concept=" against",
        criterion="top1",
        reported={"R": 4, "J": None},
        reconstructed=True,
        quote=(
            "On the misspelled word “aganst”, R-lens surfaces the correctly-spelled "
            "“against” at rank 1 at layer 4, while J-lens never surfaces “against” "
            "on the token position where the typo occurs"
        ),
    ),
    Anchor(
        name="verona-italy",
        prompt="Two households, both alike in dignity, in fair Verona where we lay our scene",
        position_token=" Verona",
        concept=" Italy",
        criterion="top1",
        reported={"R": 5, "J": None},
        reconstructed=True,
        quote=(
            "on the token “Verona”, R-lens surfaces “Italy” at rank 1 around layer 5, "
            "while it has rank >1000 with J-lens"
        ),
    ),
    Anchor(
        name="einstein-physicists",
        prompt="Albert Einstein published the theory of relativity, and the paper was read by",
        position_token=" Einstein",
        concept=" physicists",
        criterion="top10",
        reported={"R": 6, "J": None},
        reconstructed=True,
        quote=(
            "in a prompt briefly mentioning Albert Einstein, the token “ physicists” is "
            "surfaced in the top 10 by R-lens around layers 6-17 but not by J-lens"
        ),
    ),
)


def find_position(tokenizer, input_ids: list[int], token_text: str) -> int:
    """Index of the token the anchor names.

    Matches on the decoded string so it works whatever the tokenizer does with
    leading spaces; if the word splits into several pieces, the LAST piece is
    the position (that is the one carrying the assembled word's representation).
    """
    target = token_text.strip().lower()
    decoded = [tokenizer.decode([t]) for t in input_ids]
    for i in range(len(decoded) - 1, -1, -1):
        if decoded[i].strip().lower() == target:
            return i
    # fall back to the last token of a multi-piece word
    joined = ""
    for i, piece in enumerate(decoded):
        joined += piece
        if joined.strip().lower().endswith(target):
            return i
    raise ValueError(f"token {token_text!r} not found in {''.join(decoded)!r}")


def concept_ids(tokenizer, concept: str) -> list[int]:
    """Single-token surface forms of the concept (leading space, casing)."""
    ids = set()
    base = concept.strip()
    for form in {base, base.lower(), base.capitalize(),
                 " " + base, " " + base.lower(), " " + base.capitalize()}:
        pieces = tokenizer.encode(form, add_special_tokens=False)
        if len(pieces) == 1:
            ids.add(pieces[0])
    return sorted(ids)


def run_anchors(model, lenses: dict, anchors=ANCHORS, *, max_seq_len: int = 128) -> pd.DataFrame:
    """Rank of each anchor's concept at every layer, for every lens.

    Returns long form: (anchor, lens, layer, rank). Rank 1 is best; the rank is
    the best over all single-token surface forms of the concept.
    """
    import torch

    from jlens.hooks import ActivationRecorder

    jacobian = [name for name, lens in lenses.items() if lens is not None]
    if not jacobian:
        raise SystemExit("need at least one Jacobian lens; only the logit lens was loaded")
    layers = lenses[jacobian[0]].source_layers
    tok = model.tokenizer

    rows = []
    with torch.no_grad():
        for anchor in anchors:
            input_ids = model.encode(anchor.prompt, max_length=max_seq_len)
            seq = input_ids[0].tolist()
            pos = find_position(tok, seq, anchor.position_token)
            ids = concept_ids(tok, anchor.concept)
            if not ids:
                raise ValueError(
                    f"{anchor.name}: {anchor.concept!r} has no single-token form in this tokenizer"
                )

            with ActivationRecorder(model.layers, at=layers) as rec:
                model.forward(input_ids)
                acts = {l: rec.activations[l][0].detach().float() for l in layers}

            for layer in layers:
                residual = acts[layer][pos]
                for name, lens in lenses.items():
                    read = residual if lens is None else lens.transport(residual, layer)
                    logits = model.unembed(read).float()
                    best = logits[ids].max()
                    rank = int((logits > best).sum().item()) + 1
                    rows.append(
                        {"anchor": anchor.name, "lens": name, "layer": layer,
                         "rank": rank, "position": pos, "reconstructed": anchor.reconstructed}
                    )
    return pd.DataFrame(rows)


def onset(ranks: pd.DataFrame, anchor: Anchor, lens: str, *, threshold: int | None = None) -> int | None:
    """First layer at which the concept reaches ``threshold``, or None.

    Defaults to the anchor's own criterion, but both thresholds are reported:
    a rank-1 criterion transferred from the post's model can read as "never
    surfaces" on another model even when the concept clearly does surface. On
    qwen3.5-27b the `verona-italy` and `typo-aganst` anchors reach single-digit
    ranks under R-lens far earlier than under J-lens while never hitting rank 1
    — the ordering the post describes, hidden by a strict threshold.
    """
    if threshold is None:
        threshold = 1 if anchor.criterion == "top1" else 10
    if ranks.empty or "lens" not in ranks.columns:
        return None
    hit = ranks[(ranks["anchor"] == anchor.name) & (ranks["lens"] == lens) & (ranks["rank"] <= threshold)]
    return int(hit["layer"].min()) if not hit.empty else None


def verdicts(ranks: pd.DataFrame, lens_map: dict, anchors=ANCHORS) -> pd.DataFrame:
    """One row per anchor: measured onsets vs the post's reported onsets.

    ``lens_map`` maps the post's labels to our arm names, e.g.
    ``{"R": "released-R", "J": "released-J"}``.

    The verdict is DIRECTIONAL: it asks whether R surfaces the concept earlier
    than J, which is the claim that should survive a change of model. Exact
    layer agreement is reported but not required, since the post's examples are
    on its own headline model.
    """
    # An anchor whose concept never surfaces for any lens yields no rows at all;
    # that is a result ("NEITHER"), not a crash.
    present = set(ranks["lens"]) if "lens" in ranks.columns else set()
    rows = []
    for anchor in anchors:
        def at(arm, threshold):
            return onset(ranks, anchor, arm, threshold=threshold) if arm in present else None

        measured = {label: at(arm, None) for label, arm in lens_map.items()}
        top10 = {label: at(arm, 10) for label, arm in lens_map.items()}
        measured["logit"] = at("logit", None)
        # Judge on top-10: it is the threshold that survives a change of model.
        r, j = top10.get("R"), top10.get("J")
        if r is None and j is None:
            verdict = "NEITHER — concept never surfaces (check prompt/position)"
        elif r is not None and j is None:
            verdict = "MATCH — R surfaces, J never (as reported)"
        elif r is None and j is not None:
            verdict = "INVERTED — J surfaces, R never"
        elif r < j:
            verdict = f"MATCH — R earlier by {j - r} layers"
        elif r == j:
            verdict = "TIE — same onset"
        else:
            verdict = f"INVERTED — J earlier by {r - j} layers"
        rows.append(
            {
                "anchor": anchor.name,
                "reconstructed": anchor.reconstructed,
                "criterion": anchor.criterion,
                "reported_R": anchor.reported.get("R"),
                "R_top10": top10.get("R"),
                "R_top1": measured.get("R") if anchor.criterion == "top1" else at(lens_map.get("R"), 1),
                "reported_J": anchor.reported.get("J"),
                "J_top10": top10.get("J"),
                "J_top1": at(lens_map.get("J"), 1),
                "logit_top10": at("logit", 10),
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows).set_index("anchor")
