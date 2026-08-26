"""Temporal faithfulness experiment: causal concept onset (design doc §7).

Question: when the R-lens reads out an intermediate concept at an early layer,
is the concept causally present there, or is the lens merely predicting what
later layers will compute?

Three per-layer measurements at one pre-registered position t_i (final prompt
token; sensitivity: penultimate), for each lens M in {R, J, logit}:

  A. natural readout      R_l = pass@10 of c, and MRR
  B. causal necessity     N_l = drop in log p(y | x) after ablating u_c at (l, t_i)
  C. counterfactual       S_l = shift of margin log p(y') - log p(y) after the
     sufficiency               pseudoinverse coordinate swap c -> c' at (l, t_i)

Conventions (verified against the J-lens paper, the R-lens post, and
reference/idhantgulati-j-lens — see EXPERIMENT-causal-onset.md):
- swap operator: the paper's coordinate patch  z = V^+ h,  h' = h + alpha * V(Pi z - z)
  on RAW lens vectors; bisector reflection as cross-check; cos/kappa logged (§7.7.6).
- ablation: plain projection removal (no mean-centering; --center is a sensitivity).
- lens vectors: rows of (W_U diag(g_eff)) J_l with g_eff = 1 + norm.weight for the
  qwen3_5 family ((1+w)-style RMSNorm — the community repo's raw-weight version is
  wrong for this family).
- edits are post-forward-hooks on the decoder block l (the tensor J_l was fit
  against), restricted to one position and one batch element; fp32 inside.
- items: probe-swap.json (prompt, intermediate c, answer y, swap_to c', swap_answer y'),
  filtered per §7.1 against the actual model + tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from rlens.evals import rank_of, token_ids_of

REPO_ROOT = Path(__file__).resolve().parents[1]

LENSES = ("R", "J", "logit")
ALPHAS = (0.5, 1.0, 2.0)  # alpha=0 runs once as the identity/numerics check
N_RANDOM = 3


# ---------------------------------------------------------------------------
# Lens vectors
# ---------------------------------------------------------------------------

#: model_type -> effective final-RMSNorm gain. qwen3_5 (and gemma) use the
#: (1 + weight) convention; add families explicitly, never guess.
_GAIN_REGISTRY = {
    "qwen3_5": lambda norm: 1.0 + norm.weight.float(),
    "qwen3_5_text": lambda norm: 1.0 + norm.weight.float(),
}


def final_norm_gain(hf_model) -> torch.Tensor:
    model_type = hf_model.config.get_text_config().model_type
    if model_type not in _GAIN_REGISTRY:
        raise ValueError(f"no final-norm gain registered for model_type={model_type!r}")
    return _GAIN_REGISTRY[model_type](hf_model.model.norm).detach().cpu()


def lens_vectors(hf_model, J: torch.Tensor | None, token_ids: list[int]) -> torch.Tensor:
    """Raw lens vectors [n, d] (fp32, CPU): rows of (W_U diag(g_eff)) @ J.
    ``J=None`` gives the logit-lens direction (no transport)."""
    w = hf_model.lm_head.weight[list(token_ids)].detach().float().cpu()
    v = w * final_norm_gain(hf_model)
    if J is not None:
        v = v @ J.float().cpu()
    return v


# ---------------------------------------------------------------------------
# Intervention edits: pure functions on a single residual vector h [d] (fp32)
# ---------------------------------------------------------------------------


def make_ablation(u: torch.Tensor, alpha: float, mu: torch.Tensor | None = None):
    """h - alpha * <h - mu, u_hat> u_hat  (mu=None: paper convention, no centering)."""
    u_hat = (u / u.norm()).float()

    def edit(h: torch.Tensor) -> torch.Tensor:
        centered = h if mu is None else h - mu.to(h.device)
        return h - alpha * (centered @ u_hat.to(h.device)) * u_hat.to(h.device)

    return edit


def swap_diagnostics(v_s: torch.Tensor, v_t: torch.Tensor) -> dict:
    cos = float((v_s @ v_t) / (v_s.norm() * v_t.norm()))
    sv = torch.linalg.svdvals(torch.stack([v_s, v_t], dim=1).float())
    return {"cos": cos, "kappa": float(sv[0] / sv[-1])}


def make_pinv_swap(v_s: torch.Tensor, v_t: torch.Tensor, alpha: float, ridge: float = 0.0):
    """Design §7.5 / paper coordinate patch on RAW vectors:
    z = V^+ h;  h' = h + alpha * V (Pi z - z)."""
    V = torch.stack([v_s, v_t], dim=1).float()  # [d, 2]
    gram = V.T @ V

    def edit(h: torch.Tensor) -> torch.Tensor:
        Vd, Gd = V.to(h.device), gram.to(h.device).double()
        rhs = (Vd.T @ h).double()
        # fp64 2x2 solve: kills the squared conditioning of the normal equations
        # A = V^T V (+ ridge) is 2x2 positive definite; solve is exact and,
        # unlike lstsq, does not silently return garbage on CUDA for
        # near-singular input
        A = Gd + ridge * torch.eye(2, device=h.device, dtype=torch.float64)
        z = torch.linalg.solve(A, rhs)
        return h + Vd @ (alpha * (z.flip(0) - z)).float()

    return edit


def make_reflection_swap(v_s: torch.Tensor, v_t: torch.Tensor, alpha: float):
    """Community cross-check: h - 2 alpha <h, u_hat> u_hat, u_hat ∝ v_hat_s - v_hat_t.
    Identical to the pinv swap for one UNIT-normed pair at alpha=1."""
    u = v_s / v_s.norm() - v_t / v_t.norm()
    u_hat = (u / u.norm()).float()

    def edit(h: torch.Tensor) -> torch.Tensor:
        ud = u_hat.to(h.device)
        return h - 2.0 * alpha * (h @ ud) * ud

    return edit


def make_patch(donor_h: torch.Tensor):
    """Replace the activation outright with a donor run's activation.

    The only intervention in this file that involves no lens at all: it
    transports whatever the donor's residual holds at that (layer, position).
    """
    donor = donor_h.detach().clone()

    def edit(h: torch.Tensor) -> torch.Tensor:
        return donor.to(device=h.device, dtype=h.dtype).float()

    return edit


def make_random_displacement(delta_norm: float, d: int, seed: int):
    """h + g with ||g|| matched to a real intervention's displacement (§7.7.2)."""
    g = torch.randn(d, generator=torch.Generator().manual_seed(seed))
    g = g / g.norm() * delta_norm

    def edit(h: torch.Tensor) -> torch.Tensor:
        return h + g.to(h.device)

    return edit


# ---------------------------------------------------------------------------
# Batched edited forward
# ---------------------------------------------------------------------------


class EditRunner:
    """Run one prompt under many position-local edits in one batch.

    Each batch element is a ``{layer: edit_fn}`` dict — usually one layer
    (layer-local interventions), or several for the persistent anti-self-repair
    ablation. Post-forward hooks on the decoder blocks (same tensors the lens
    was fit against); element b's edit for layer l is applied to h[b, position]
    in fp32. Returns final-position logits [n_edits, vocab] (fp32, CPU).
    """

    def __init__(self, hf_model):
        self.hf_model = hf_model
        self.layers = hf_model.model.layers

    @torch.no_grad()
    def run(self, input_ids: torch.Tensor, position: int, edits: list[dict], batch_size: int = 24) -> torch.Tensor:
        out = []
        for start in range(0, len(edits), batch_size):
            chunk = list(edits[start : start + batch_size])
            # pad to a constant batch shape: cuBLAS kernel choice (and hence
            # bf16 rounding) depends on shape, so all chunks must match the
            # chunk containing the identity/clean reference exactly
            n_real = len(chunk)
            if n_real < batch_size and len(edits) > batch_size:
                chunk += [chunk[-1]] * (batch_size - n_real)
            batch = input_ids.expand(len(chunk), -1)

            def make_hook(layer_idx, chunk=chunk):
                def hook(module, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    for b, per_layer in enumerate(chunk):
                        edit = per_layer.get(layer_idx)
                        if edit is not None:
                            h[b, position] = edit(h[b, position].float()).to(h.dtype)
                    return output

                return hook

            hooked = sorted({l for per_layer in chunk for l in per_layer})
            handles = [self.layers[l].register_forward_hook(make_hook(l)) for l in hooked]
            try:
                logits = self.hf_model(input_ids=batch, use_cache=False).logits
            finally:
                for handle in handles:
                    handle.remove()
            out.append(logits[:n_real, -1].float().cpu())
        return torch.cat(out)


# ---------------------------------------------------------------------------
# Dataset (§7.1)
# ---------------------------------------------------------------------------


@dataclass
class OnsetItem:
    name: str
    category: str
    prompt: str            # rstripped
    control_prompt: str
    c: str
    c_prime: str
    c_wrong: str
    y: str
    y_prime: str
    c_id: int              # chosen single-token id for the direction
    c_prime_id: int
    c_wrong_id: int
    y_id: int              # chosen by clean-logit argmax over surface forms
    y_prime_id: int        # chosen under the CONTROL prompt (where y' is the answer)
    t_i: int               # absolute intervention/readout position
    cue_i: int             # cue-token position (the post's flagship readouts live here)
    c_wrong_same_category: bool = True   # realized quality of the §7.7.3 control
    shuffled_c: str = ""   # readout baseline concept (different category, seeded)


#: pre-registered stopword list for the cue auto-rule
_CUE_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "with", "is",
    "was", "are", "were", "that", "which", "where", "who", "whose", "its", "it",
    "and", "or", "most", "from", "into", "s", "called", "fact", "used", "when",
}


def find_cue(token_strs: list[str], override_word: str | None = None) -> int:
    """Cue position: the descriptor's content token (e.g. 'sushi', 'boot').

    Rule (pre-registered): the override word's last token if given; else the
    last mid-sentence Capitalized token (proper nouns like 'Amazon'); else the
    last alphabetic non-stopword token. Audit the choice via the data stage's
    printout and hand-fix misses in data/onset_cues.json.
    """
    stripped = [t.strip() for t in token_strs]
    if override_word is not None:
        w = override_word.lower()
        for i in range(len(stripped) - 1, -1, -1):
            t = stripped[i].lower()
            # match whole word in one token, or a subtoken of a multi-token
            # word (>=3 chars, or a digit string like atomic numbers)
            if t and (w in t or (t in w and (len(t) >= 3 or t.isdigit()))):
                return i
    for i in range(len(stripped) - 1, 0, -1):
        t = stripped[i]
        if t.isalpha() and t[0].isupper() and t.lower() not in _CUE_STOPWORDS:
            return i
    for i in range(len(stripped) - 1, -1, -1):
        t = stripped[i].lower()
        if t.isalpha() and t not in _CUE_STOPWORDS:
            return i
    return len(stripped) - 1


def _stable_seed(*parts) -> int:
    """Process-independent seed (builtin hash() is salted per process, which
    made the random controls irreproducible across runs)."""
    digest = hashlib.blake2b("\x1f".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def _direction_id(tok, word: str) -> int | None:
    """One token id for a concept's direction: leading-space form preferred."""
    for form in (f" {word}", word, f" {word.lower()}", word.lower(), f" {word.capitalize()}"):
        ids = tok.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


#: control-prompt construction: "The <prop> <prep> the <descriptor> is" ->
#: "Fact: The <prop> <prep> <c'> is". Items matching no pattern are dropped
#: (logged) unless a hand-written control exists in data/onset_controls.json.
_CONTROL_PATTERNS = [
    re.compile(r"^(?:Fact: )?(?P<head>The .+? (?:of|in|on|for|used in|used by|spoken in|played in|found in|grown in|written in|celebrated in)) the .+? (?P<tail>is(?: called)?(?: the)?(?: a)?)$"),
    re.compile(r"^(?:Fact: )?(?P<head>The .+? (?:of|in|on)) the .+? (?P<tail>is)$"),
]


def build_control_prompt(prompt: str, c_prime: str, overrides: dict) -> str | None:
    if prompt in overrides:
        return overrides[prompt]
    for pattern in _CONTROL_PATTERNS:
        m = pattern.match(prompt)
        if m:
            return f"Fact: {m.group('head')} {c_prime} {m.group('tail')}"
    return None


@torch.no_grad()
def build_dataset(hf_model, tok, *, position: int = -1, min_p_answer: float = 1e-3, limit: int | None = None) -> tuple[list[OnsetItem], list[dict]]:
    """Filter probe-swap items per §7.1 against the actual model + tokenizer."""
    raw = json.loads(
        (REPO_ROOT / "data" / "eval_prompts" / "experiments" / "probe-swap.json").read_text(encoding="utf-8")
    )["items"][:limit]
    overrides_path = REPO_ROOT / "data" / "onset_controls.json"
    overrides = json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
    cues_path = REPO_ROOT / "data" / "onset_cues.json"
    cue_overrides = json.loads(cues_path.read_text(encoding="utf-8")) if cues_path.exists() else {}

    device = hf_model.model.embed_tokens.weight.device

    def top1_and_logp(prompt: str, id_set: list[int]) -> tuple[bool, float, torch.Tensor]:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        logits = hf_model(input_ids=ids, use_cache=False).logits[0, -1].float().cpu()
        logp = logits.log_softmax(-1)
        return int(logits.argmax()) in id_set, float(logp[id_set].max()), logp

    # wrong-concept: the NEXT same-category item's c' (cyclic), else next item's c
    by_cat: dict[str, list[dict]] = {}
    for item in raw:
        by_cat.setdefault(item["category"], []).append(item)

    kept, log = [], []

    def drop(item, reason):
        log.append({"name": item["name"], "kept": False, "reason": reason})

    for item in raw:
        prompt = item["prompt"].rstrip()
        c, c_prime, y, y_prime = item["intermediate"], item["swap_to"], item["answer"], item["swap_answer"]
        if y.lower() == y_prime.lower():
            drop(item, "y == y'"); continue
        ids = {k: _direction_id(tok, v) for k, v in {"c": c, "c'": c_prime, "y": y, "y'": y_prime}.items()}
        if any(v is None for v in ids.values()):
            drop(item, f"multi-token: {[k for k, v in ids.items() if v is None]}"); continue
        # word-boundary, not substring: "Oman" must not match "Roman"
        if any(re.search(rf"\b{re.escape(w.lower())}\b", prompt.lower()) for w in (c, c_prime)):
            drop(item, "c or c' verbatim in prompt"); continue

        y_set, yp_set = token_ids_of(tok, y), token_ids_of(tok, y_prime)
        correct, logp_y, logp = top1_and_logp(prompt, y_set)
        if not correct:
            drop(item, "clean answer wrong"); continue
        if logp_y < torch.log(torch.tensor(min_p_answer)):
            drop(item, f"p(y) < {min_p_answer}"); continue

        control = build_control_prompt(prompt, c_prime, overrides)
        if control is None:
            drop(item, "no control template (add to data/onset_controls.json)"); continue
        control_ok, _, control_logp = top1_and_logp(control, yp_set)
        if not control_ok:
            drop(item, "control answer wrong"); continue

        # Wrong-concept control (§7.7.3): must be SAME category — the control
        # asks "does any concept of this kind do this?". Singleton categories
        # fall back to a per-item *shuffled* global pool (never a fixed token:
        # the old first-match-wins fallback handed 52% of items the same word).
        cat_items = [o for o in by_cat[item["category"]] if o is not item]
        same_cat_pool = [w for o in cat_items for w in (o["intermediate"], o["swap_to"])]
        rng_item = random.Random(_stable_seed(item["name"], "wrong"))
        global_pool = [w for o in raw for w in (o["intermediate"], o["swap_to"])]
        rng_item.shuffle(global_pool)
        excluded = {c.lower(), c_prime.lower()}

        def pick(pool):
            return next((w for w in pool if w.lower() not in excluded and _direction_id(tok, w) is not None), None)

        c_wrong = pick(same_cat_pool)
        same_category = c_wrong is not None
        if c_wrong is None:
            c_wrong = pick(global_pool)
        if c_wrong is None:
            drop(item, "no wrong-concept candidate"); continue

        token_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        seq_len = len(token_ids)
        token_strs = [tok.decode([t]) for t in token_ids]
        cue_i = find_cue(token_strs, cue_overrides.get(prompt))
        y_id = y_set[int(logp[y_set].argmax())]
        # y' surface form is chosen where y' is actually the answer (the
        # control prompt), not under the clean prompt where it never appears
        yp_id = yp_set[int(control_logp[yp_set].argmax())]
        kept.append(
            OnsetItem(
                name=item["name"], category=item["category"], prompt=prompt,
                control_prompt=control, c=c, c_prime=c_prime, c_wrong=c_wrong,
                y=y, y_prime=y_prime,
                c_id=ids["c"], c_prime_id=ids["c'"], c_wrong_id=_direction_id(tok, c_wrong),
                y_id=y_id, y_prime_id=yp_id,
                t_i=seq_len + position if position < 0 else position,
                cue_i=cue_i, c_wrong_same_category=same_category,
            )
        )
        log.append({"name": item["name"], "kept": True, "cue": token_strs[cue_i],
                    "c_wrong": c_wrong, "c_wrong_same_category": same_category})

    # readout baseline: a concept from a DIFFERENT category, seeded per item
    # (the old items[(i+1) % n] neighbour was same-category 32.7% of the time)
    for it in kept:
        others = [o for o in kept if o.category != it.category] or [o for o in kept if o is not it]
        it.shuffled_c = random.Random(_stable_seed(it.name, "shuffled")).choice(others).c
    return kept, log


# ---------------------------------------------------------------------------
# Measurements (§7.3-7.5, §7.7)
# ---------------------------------------------------------------------------


def _layer_jacobians(lenses: dict, layer: int) -> dict[str, torch.Tensor | None]:
    """Per-layer J for each lens name; the logit lens has no transport."""
    return {"R": lenses["R"].jacobians[layer], "J": lenses["J"].jacobians[layer], "logit": None}


def _edit_battery(
    hf_model, jacobians: dict[str, torch.Tensor | None], item: OnsetItem, *,
    layer: int, dirs: dict[str, dict[str, dict[int, torch.Tensor]]], all_layers: list[int],
    ridge: float, center_mu: torch.Tensor | None,
):
    """All (lens, condition, alpha) edits for one (item, layer) as
    ``{layer: edit_fn}`` dicts. ``dirs[token][lens][l]`` are per-layer
    directions (token in {"c", "wrong", "y"}) for the multi-layer conditions."""
    labels, edits = [], []

    def add(lens, condition, alpha, edit_dict):
        labels.append({"lens": lens, "condition": condition, "alpha": alpha})
        edits.append(edit_dict)

    def multi(token, lens, layer_set):
        return {l2: make_ablation(dirs[token][lens][l2], 1.0, mu=center_mu) for l2 in layer_set}

    def multi_swap(lens, layer_set):
        """Band swap: the pinv swap applied at every layer in the set, each
        with that layer's own directions. The anti-self-repair counterpart of
        the swap, and the shape of the released 'swap across the band'
        protocol. NB persist (l->end) is maximal at l=0 by construction, so
        the band (0->l) is the correct repair-robust ONSET direction."""
        return {
            l2: make_pinv_swap(dirs["c"][lens][l2], dirs["c_prime"][lens][l2], 1.0, ridge=ridge)
            for l2 in layer_set
        }

    add("-", "identity", 0.0, {layer: lambda h: h})  # alpha=0 numerics check
    later = [l2 for l2 in all_layers if l2 >= layer]
    upto = [l2 for l2 in all_layers if l2 <= layer]
    for lens in LENSES:
        J = jacobians[lens]
        v_c, v_cp, v_wrong, v_y, v_yp = lens_vectors(
            hf_model, J, [item.c_id, item.c_prime_id, item.c_wrong_id, item.y_id, item.y_prime_id]
        )
        add(lens, "ablate", 1.0, {layer: make_ablation(v_c, 1.0, mu=center_mu)})
        # persistent ablation (anti-self-repair): suppress the concept at every
        # layer >= l. band ablation: the complement, layers 0..l — the pair
        # isolates the early-band causal contribution.
        add(lens, "ablate_persist", 1.0, multi("c", lens, later))
        add(lens, "ablate_band", 1.0, multi("c", lens, upto))
        # collateral-damage controls for the persist curve: if suppressing an
        # unrelated concept / the answer direction from the same starting layer
        # matches, late-band suppression is generic answer-space damage.
        add(lens, "ablate_wrong_persist", 1.0, multi("wrong", lens, later))
        add(lens, "ablate_answer_persist", 1.0, multi("y", lens, later))
        add(lens, "ablate_wrong", 1.0, {layer: make_ablation(v_wrong, 1.0, mu=center_mu)})
        add(lens, "ablate_answer", 1.0, {layer: make_ablation(v_y, 1.0, mu=center_mu)})
        for alpha in ALPHAS:
            add(lens, "swap", alpha, {layer: make_pinv_swap(v_c, v_cp, alpha, ridge=ridge)})
        add(lens, "swap_band", 1.0, multi_swap(lens, upto))
        add(lens, "swap_reflection", 1.0, {layer: make_reflection_swap(v_c, v_cp, 1.0)})
        add(lens, "swap_wrong", 1.0, {layer: make_pinv_swap(v_wrong, v_cp, 1.0, ridge=ridge)})
        add(lens, "swap_answer", 1.0, {layer: make_pinv_swap(v_y, v_yp, 1.0, ridge=ridge)})
    return labels, edits


@torch.no_grad()
def run_measurements(
    hf_model, tok, lenses: dict[str, "object"], items: list[OnsetItem], *,
    layers: list[int], ridge: float = 0.0, center: bool = False, batch_size: int = 24,
    intervene_at: str = "t",  # "t" = pre-registered t_i; "cue" = the cue token
) -> pd.DataFrame:
    """One record per (item, layer, lens, condition, alpha) with margins/logps,
    plus readout rows (condition='readout') and clean rows per item.

    ``lenses`` maps 'R'/'J' to loaded JacobianLens objects ('logit' derived).
    """
    from jlens.hooks import ActivationRecorder

    device = hf_model.model.embed_tokens.weight.device
    runner = EditRunner(hf_model)
    records: list[dict] = []

    def unembed(h: torch.Tensor) -> torch.Tensor:
        w = hf_model.lm_head.weight
        return hf_model.lm_head(hf_model.model.norm(h.to(w.device, w.dtype))).float().cpu()

    mu_by_layer: dict[int, torch.Tensor] = {}
    if center:  # calibration means over the items' own clean activations (pass 1)
        sums, count = {l: None for l in layers}, 0
        for item in items:
            ids = tok(item.prompt, return_tensors="pt").input_ids.to(device)
            with ActivationRecorder(hf_model.model.layers, at=layers) as rec:
                hf_model(input_ids=ids, use_cache=False)
                for l in layers:
                    mu_pos = item.cue_i if intervene_at == "cue" else item.t_i
                    h = rec.activations[l][0, mu_pos].float().cpu()
                    sums[l] = h if sums[l] is None else sums[l] + h
            count += 1
        mu_by_layer = {l: sums[l] / count for l in layers}

    for item_idx, item in enumerate(items):
        input_ids = tok(item.prompt, return_tensors="pt").input_ids.to(device)
        with ActivationRecorder(hf_model.model.layers, at=layers) as rec:
            clean_logits = hf_model(input_ids=input_ids, use_cache=False).logits[0, -1].float().cpu()
            acts = {l: rec.activations[l][0].detach() for l in layers}
        clean_logp = clean_logits.log_softmax(-1)
        clean_margin = float(clean_logp[item.y_prime_id] - clean_logp[item.y_id])
        base = {
            "item": item.name, "category": item.category,
            "clean_logp_y": float(clean_logp[item.y_id]), "clean_margin": clean_margin,
            # batch-of-1 values kept for the numerics report; intervention rows
            # override the non-_single keys with the in-batch identity reference
            "clean_logp_y_single": float(clean_logp[item.y_id]), "clean_margin_single": clean_margin,
        }

        # shuffled-label readout baseline: a different-category concept fixed
        # per item at dataset-build time (falls back to the cyclic neighbour
        # for datasets built before this field existed)
        shuffled_c = getattr(item, "shuffled_c", "") or items[(item_idx + 1) % len(items)].c

        # per-layer directions for c / wrong-concept / answer (multi-layer
        # conditions), once per item
        token_ids_by_key = {"c": item.c_id, "c_prime": item.c_prime_id,
                            "wrong": item.c_wrong_id, "y": item.y_id}
        dirs = {key: {lens: {} for lens in LENSES} for key in token_ids_by_key}
        for l in layers:
            per_layer_J = _layer_jacobians(lenses, l)
            for lens in LENSES:
                vs = lens_vectors(hf_model, per_layer_J[lens], list(token_ids_by_key.values()))
                for key, v in zip(token_ids_by_key, vs):
                    dirs[key][lens][l] = v

        # lens-geometry diagnostics: if R's and J's concept directions are
        # near-parallel, identical results are forced by construction
        for l in layers:
            vR, vJ, vL = (dirs["c"][lens][l] for lens in ("R", "J", "logit"))
            records.append({
                **base, "layer": l, "lens": "-", "condition": "lens_cos", "alpha": 0.0,
                "cos_RJ": float((vR @ vJ) / (vR.norm() * vJ.norm())),
                "cos_Rlogit": float((vR @ vL) / (vR.norm() * vL.norm())),
            })
            # per-(layer, item) swap-pair conditioning (§7.7.6) + the within-lens
            # norm ratio, which is the confound the pinv swap is actually
            # sensitive to (it is invariant to rescaling BOTH vectors)
            for lens in LENSES:
                v_c, v_cp = dirs["c"][lens][l], dirs["c_prime"][lens][l]
                records.append({
                    **base, "layer": l, "lens": lens, "condition": "diagnostics", "alpha": 0.0,
                    **swap_diagnostics(v_c, v_cp),
                    "norm_ratio_cp_c": float(v_cp.norm() / v_c.norm()),
                })

        pos_i = item.cue_i if intervene_at == "cue" else item.t_i
        for layer in layers:
            h_t = acts[layer][pos_i].float().cpu()  # intervention-site activation
            h_final = acts[layer][item.t_i].float().cpu()
            h_cue = acts[layer][item.cue_i].float().cpu()
            jacobians = _layer_jacobians(lenses, layer)
            for lens in LENSES:
                J = jacobians[lens]
                for h_pos, suffix in ((h_final, ""), (h_cue, "_cue")):
                    read = h_pos if J is None else h_pos @ J.float().T
                    logits_l = unembed(read)
                    for concept, cond in ((item.c, f"readout{suffix}"), (shuffled_c, f"readout_shuffled{suffix}")):
                        ids = token_ids_of(tok, concept)
                        records.append({**base, "layer": layer, "lens": lens, "condition": cond, "alpha": 0.0, "rank_c": rank_of(logits_l, ids)})

            labels, edits = _edit_battery(
                hf_model, jacobians, item, layer=layer, dirs=dirs, all_layers=layers,
                ridge=ridge, center_mu=mu_by_layer.get(layer),
            )
            # cross-lens norm-equalized variants (magnitude confound: early-band
            # ||dh_R||/||dh_J|| measured at 3-4x): same directions, displacement
            # rescaled to the R/J geometric-mean norm at this (item, layer)
            for cond in ("ablate", "swap"):
                base_edits = {
                    lab["lens"]: ed[layer] for lab, ed in zip(labels, edits)
                    if lab["condition"] == cond and lab["alpha"] == 1.0 and lab["lens"] in ("R", "J")
                }
                delta = {lens: ed(h_t.clone()) - h_t for lens, ed in base_edits.items()}
                target = float((delta["R"].norm() * delta["J"].norm()).sqrt())
                for lens in ("R", "J"):
                    norm = float(delta[lens].norm())
                    vec = delta[lens] * (target / norm) if norm > 1e-8 else delta[lens]
                    labels.append({"lens": lens, "condition": f"{cond}_eqnorm", "alpha": 1.0, "delta_norm": target})
                    edits.append({layer: (lambda h, v=vec: h + v.to(h.device))})
            # displacement norms at this layer (magnitude-confound reporting:
            # per-lens intervention sizes differ, and effects must be judged
            # against equally-sized random pushes)
            deltas = {}
            for label, edit_dict in zip(labels, edits):
                if layer in edit_dict:
                    label["delta_norm"] = float((edit_dict[layer](h_t.clone()) - h_t).norm())
                if label["condition"] in ("ablate", "swap") and label["alpha"] == 1.0:
                    deltas[(label["lens"], label["condition"])] = label["delta_norm"]
            # norm-matched random controls for the real ablation and swap(1) of each lens
            for (lens, cond), dn in deltas.items():
                for b in range(N_RANDOM):
                    labels.append({"lens": lens, "condition": f"random_{cond}", "alpha": 1.0, "delta_norm": dn})
                    edits.append({layer: make_random_displacement(
                        dn, h_t.shape[0], seed=_stable_seed(item.name, layer, lens, cond, b))})

            # NB: pos_i, not item.t_i — the edits must be applied at the
            # intervention site. (Edit fns are built from lens vectors, so a
            # wrong position here silently reproduces the t_i run exactly.)
            logits = runner.run(input_ids, pos_i, edits, batch_size=batch_size)
            logp = logits.log_softmax(-1)
            # effects are referenced to the IN-BATCH identity condition
            # (labels[0]): identical kernel shapes -> bf16 shape-noise cancels
            # exactly. The batch-of-1 clean values stay as *_single for the
            # numerics report.
            clean_b_logp_y = float(logp[0][item.y_id])
            clean_b_margin = float(logp[0][item.y_prime_id] - logp[0][item.y_id])
            for label, lp, lg in zip(labels, logp, logits):
                records.append({
                    **base, "layer": layer, **label,
                    "clean_logp_y": clean_b_logp_y, "clean_margin": clean_b_margin,
                    "logp_y": float(lp[item.y_id]), "logp_yp": float(lp[item.y_prime_id]),
                    "margin": float(lp[item.y_prime_id] - lp[item.y_id]),
                    "top1_is_yp": int(int(lg.argmax()) in token_ids_of(tok, item.y_prime)),
                    "top1_is_y": int(int(lg.argmax()) in token_ids_of(tok, item.y)),
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Lens-free ceiling: cross-prompt activation patching at the cue
# ---------------------------------------------------------------------------


def build_patch_pairs(items: list[OnsetItem], tok) -> list[tuple[OnsetItem, OnsetItem]]:
    """Ordered (receiver, donor) pairs from token-identical prompt templates.

    Two prompts qualify when they differ in exactly one token — the cue — so the
    patch transports the cue's meaning without any structural mismatch. Pairs
    with the same answer or the same intermediate are dropped (nothing to
    detect).
    """
    groups: dict[tuple, list[OnsetItem]] = {}
    for item in items:
        ids = tok(item.prompt, return_tensors="pt").input_ids[0].tolist()
        key = tuple("<CUE>" if i == item.cue_i else t for i, t in enumerate(ids))
        groups.setdefault(key, []).append(item)

    pairs = []
    for members in groups.values():
        for recv in members:
            for donor in members:
                if donor is recv or recv.y == donor.y or recv.c == donor.c:
                    continue
                assert recv.cue_i == donor.cue_i, "template group with mismatched cue index"
                pairs.append((recv, donor))
    return pairs


@torch.no_grad()
def run_patching(
    hf_model, tok, items: list[OnsetItem], *, layers: list[int], batch_size: int = 24,
    filler_offset: int = 2, patch_at: str = "final",
) -> pd.DataFrame:
    """Sweep the patch layer for every (receiver, donor) pair.

    ``patch_at="final"`` (default, the informative one): patch the FINAL prompt
    token. Template-matched pairs share that token exactly ("... is"), so the
    patch transports only *computed* state -- l* is then a genuine onset for
    "by layer l the final position encodes enough to produce the donor's
    answer".

    ``patch_at="cue"``: patch the cue token. Beware -- at early layers the cue
    position still *is* the cue token, so patching it is equivalent to editing
    the prompt and downstream simply recomputes. That makes l* trivially 0; the
    cue sweep is informative only for its OFFSET (the layer by which the cue
    stops being decisive, i.e. the information has moved on).

    Metrics: ``top1_is_donor`` (primary -- directional, only satisfied by
    installing the donor's answer) and the swap-comparable margin
    log p(y_donor) - log p(y_recv) (secondary: it also rewards merely
    destroying the receiver's answer, which the random control does).

    Conditions: ``patch``, ``patch_selfpatch`` (donor == receiver, no-op check),
    ``patch_filler`` (donor vector at a different position), ``patch_random``
    (norm-matched random vector).
    """
    from jlens.hooks import ActivationRecorder

    device = hf_model.model.embed_tokens.weight.device
    runner = EditRunner(hf_model)
    pairs = build_patch_pairs(items, tok)

    def site_of(item: OnsetItem) -> int:
        return item.cue_i if patch_at == "cue" else item.t_i

    # cache each item's clean residual at the patch site (and at a control
    # position). Recorded with the same batch shape the patched forwards use,
    # so bf16 kernel-shape noise cancels in the self-patch check.
    cache: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    for item in {p.name: p for pair in pairs for p in pair}.values():
        input_ids = tok(item.prompt, return_tensors="pt").input_ids.to(device)
        site_i = site_of(item)
        filler_i = max(0, site_i - filler_offset)
        with ActivationRecorder(hf_model.model.layers, at=layers) as rec:
            hf_model(input_ids=input_ids.expand(batch_size, -1), use_cache=False)
            cache[item.name] = {
                "cue": {l: rec.activations[l][0, site_i].detach().float().cpu() for l in layers},
                "filler": {l: rec.activations[l][0, filler_i].detach().float().cpu() for l in layers},
                "filler_i": filler_i,
            }

    records = []
    for recv, donor in pairs:
        input_ids = tok(recv.prompt, return_tensors="pt").input_ids.to(device)
        base = {"receiver": recv.name, "donor": donor.name, "category": recv.category,
                "y_recv": recv.y, "y_donor": donor.y, "patch_at": patch_at}
        for layer in layers:
            d_cue = cache[donor.name]["cue"][layer]
            labels = [
                ("identity", {}),
                ("patch", {layer: make_patch(d_cue)}),
                ("patch_selfpatch", {layer: make_patch(cache[recv.name]["cue"][layer])}),
                ("patch_random", {layer: make_patch(
                    torch.randn(d_cue.shape[0], generator=torch.Generator().manual_seed(
                        _stable_seed(recv.name, donor.name, layer))) * (d_cue.norm() / (d_cue.shape[0] ** 0.5)))}),
            ]
            # position-specificity control: same donor vector, non-cue position
            filler_edits = {layer: make_patch(cache[donor.name]["filler"][layer])}

            logits = runner.run(input_ids, site_of(recv), [e for _, e in labels], batch_size=batch_size)
            filler_logits = runner.run(input_ids, cache[recv.name]["filler_i"], [filler_edits], batch_size=batch_size)

            logp = logits.log_softmax(-1)
            ref = float(logp[0][donor.y_id] - logp[0][recv.y_id])  # in-batch identity reference
            for (name, _), lp, lg in zip(labels, logp, logits):
                records.append({**base, "layer": layer, "condition": name,
                                "margin": float(lp[donor.y_id] - lp[recv.y_id]), "clean_margin": ref,
                                "top1_is_donor": int(int(lg.argmax()) in token_ids_of(tok, donor.y))})
            flp = filler_logits.log_softmax(-1)[0]
            records.append({**base, "layer": layer, "condition": "patch_filler",
                            "margin": float(flp[donor.y_id] - flp[recv.y_id]), "clean_margin": ref,
                            "top1_is_donor": int(int(filler_logits[0].argmax()) in token_ids_of(tok, donor.y))})
    return pd.DataFrame(records)


def analyze_patching(df: pd.DataFrame, *, rho: float = 0.2, w: int = 2, n_boot: int = 500, seed: int = 0) -> dict:
    """l* = first layer where patching the cue transfers the donor's answer."""
    df = df.copy()
    df["effect"] = df["margin"] - df["clean_margin"]
    pairs = sorted({(r, d) for r, d in zip(df["receiver"], df["donor"])})
    rng = torch.Generator().manual_seed(seed)

    def curves(sub: pd.DataFrame) -> dict[str, pd.Series]:
        by = lambda c, col="effect": sub[sub.condition == c].groupby("layer")[col].mean()
        return {"patch": by("patch"), "filler": by("patch_filler"),
                "random": by("patch_random"), "selfpatch": by("patch_selfpatch"),
                # directional: only satisfied by installing the DONOR's answer.
                # The margin alone also rewards destroying the receiver's answer,
                # which is exactly what a random vector does.
                "flip": by("patch", "top1_is_donor"), "flip_random": by("patch_random", "top1_is_donor")}

    def l_star(cv: dict) -> float | None:
        return _onset(cv["flip"] - cv["flip_random"], rho, w)

    full = curves(df)
    boot = []
    for _ in range(n_boot):
        pick = [pairs[i] for i in torch.randint(len(pairs), (len(pairs),), generator=rng).tolist()]
        sub = pd.concat([df[(df.receiver == r) & (df.donor == d)] for r, d in pick])
        v = l_star(curves(sub))
        if v is not None:
            boot.append(v)
    s = pd.Series(boot)
    return {
        "curves": full, "l_star": l_star(full),
        "l_star_ci": (float(s.quantile(0.025)), float(s.quantile(0.975))) if len(s) else None,
        "defined_frac": len(boot) / max(n_boot, 1), "n_pairs": len(pairs),
        "top1_flip_rate": float(df[df.condition == "patch"]["top1_is_donor"].mean()),
        # per-layer MEAN drift is the meaningful check; a record-level max is
        # dominated by bf16 tails on single (pair, layer) cells
        "selfpatch_mean_drift": float(full["selfpatch"].abs().mean()),
        "selfpatch_max_layer_drift": float(full["selfpatch"].abs().max()),
        "patch_at": (df["patch_at"].iloc[0] if "patch_at" in df else "cue"),
        # the offset: last layer where patching still flips the answer
        "l_offset": (float(full["flip"][full["flip"] > 0.5 * full["flip"].max()].index.max())
                     if float(full["flip"].max()) > 0 else None),
    }


# ---------------------------------------------------------------------------
# Analysis (§7.8-7.9)
# ---------------------------------------------------------------------------


def _onset(curve: pd.Series, rho: float, w: int = 2) -> float | None:
    """First layer whose normalized excess stays >= rho for w layers.

    NaN-safe: a missing layer must not poison the whole curve. (With plain
    ``.max()`` one NaN makes ``peak`` NaN, every comparison False, and the
    function returns None — an "undefined onset" caused by a gap in the data
    rather than by the data itself.)
    """
    import numpy as np

    excess = curve.to_numpy(dtype=float)
    peak = np.nanmax(excess) if np.any(np.isfinite(excess)) else np.nan
    if not np.isfinite(peak) or peak <= 0:
        return None
    norm = excess / (peak + 1e-9)
    layers = curve.index.to_numpy()
    for i in range(len(norm) - w + 1):
        window = norm[i : i + w]
        if np.all(np.isfinite(window)) and np.all(window >= rho):
            return float(layers[i])
    return None


def analyze(df: pd.DataFrame, *, rho: float = 0.2, w: int = 2, n_boot: int = 1000, seed: int = 0) -> dict:
    """Excess curves, onsets, gaps, and paired bootstrap CI for Delta_R - Delta_J."""
    items = sorted(df["item"].unique())
    rng = torch.Generator().manual_seed(seed)

    df = df.copy()
    df["hit10"] = (df["rank_c"] <= 10).astype(float) if "rank_c" in df else float("nan")
    df["necessity"] = df["clean_logp_y"] - df["logp_y"]      # N_l,i (eq. 17)
    df["swap_effect"] = df["margin"] - df["clean_margin"]    # S_l,i (eq. 23)

    def curves(sub: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
        out = {}
        for lens in LENSES:
            d = sub[sub.lens == lens]

            def by_layer(cond: str, col: str, alpha: float | None = None) -> pd.Series:
                m = d.condition == cond
                if alpha is not None:
                    m &= d.alpha == alpha
                return d[m].groupby("layer")[col].mean()

            rand_abl = by_layer("random_ablate", "necessity")
            rand_swap = by_layer("random_swap", "swap_effect")
            out[lens] = {
                "R": by_layer("readout", "hit10") - by_layer("readout_shuffled", "hit10"),
                "R_cue": by_layer("readout_cue", "hit10") - by_layer("readout_shuffled_cue", "hit10"),
                "N": by_layer("ablate", "necessity", 1.0) - rand_abl,
                "S": by_layer("swap", "swap_effect", 1.0) - rand_swap,
                # magnitude-controlled (cross-lens equalized displacement)
                "N_eqnorm": by_layer("ablate_eqnorm", "necessity", 1.0) - rand_abl,
                "S_eqnorm": by_layer("swap_eqnorm", "swap_effect", 1.0) - rand_swap,
                # repair-robust: intervene at every layer 0..l
                "N_band": by_layer("ablate_band", "necessity", 1.0) - rand_abl,
                "S_band": by_layer("swap_band", "swap_effect", 1.0) - rand_swap,
                # informational only — persist (l->end) is maximal at l=0 by
                # construction, so it is NOT an onset statistic
                "N_persist": by_layer("ablate_persist", "necessity", 1.0) - rand_abl,
                "N_wrong_persist": by_layer("ablate_wrong_persist", "necessity", 1.0) - rand_abl,
                "N_answer_persist": by_layer("ablate_answer_persist", "necessity", 1.0) - rand_abl,
            }
        return out

    #: onsets computed for every variant; L_causal is derived three ways
    ONSET_KEYS = ("R", "R_cue", "N", "S", "N_eqnorm", "S_eqnorm", "N_band", "S_band")
    #: (variant name, necessity curve, sufficiency curve) — raw is the
    #: pre-registered primary; the other two must agree for the result to hold
    VARIANTS = (("raw", "N", "S"), ("eqnorm", "N_eqnorm", "S_eqnorm"), ("band", "N_band", "S_band"))

    def _joint(cv_lens: dict, n_key: str, s_key: str) -> float | None:
        """First layer where BOTH necessity and sufficiency stay above rho."""
        import numpy as np

        n_curve, s_curve = cv_lens[n_key], cv_lens[s_key]
        n_peak, s_peak = np.nanmax(n_curve.to_numpy(dtype=float)), np.nanmax(s_curve.to_numpy(dtype=float))
        if not (np.isfinite(n_peak) and np.isfinite(s_peak)) or n_peak <= 0 or s_peak <= 0:
            return None
        ok = ((n_curve / (n_peak + 1e-9) >= rho) & (s_curve / (s_peak + 1e-9) >= rho)).to_numpy()
        for i in range(len(ok) - w + 1):
            if ok[i : i + w].all():
                return float(n_curve.index[i])
        return None

    def onsets(cv: dict) -> dict:
        res = {}
        for lens in LENSES:
            r = {f"L_{k}": _onset(cv[lens][k], rho, w) for k in ONSET_KEYS}
            for name, n_key, s_key in VARIANTS:
                joint = _joint(cv[lens], n_key, s_key)
                suffix = "" if name == "raw" else f"_{name}"
                r[f"L_causal{suffix}"] = joint
                # gap: naming -> usable. gap_cue is the apples-to-apples one for
                # cue-site interventions; gap uses the final-token readout and
                # therefore MIXES positions (kept for continuity, do not headline).
                r[f"gap{suffix}"] = (joint - r["L_R"]) if (joint is not None and r["L_R"] is not None) else None
                r[f"gap_cue{suffix}"] = (joint - r["L_R_cue"]) if (joint is not None and r["L_R_cue"] is not None) else None
            res[lens] = r
        return res

    full_curves = curves(df)
    full_onsets = onsets(full_curves)

    # every gap flavour gets its own bootstrap distribution; the gap_cue* keys
    # are the apples-to-apples quantities for cue-site interventions
    gap_keys = [k for k in full_onsets["R"] if k.startswith("gap")]
    onset_keys = [k for k in full_onsets["R"] if k.startswith("L_")]
    boot_diffs = {k: [] for k in gap_keys}
    boot_onsets = {lens: {k: [] for k in onset_keys} for lens in LENSES}
    for _ in range(n_boot):
        pick = torch.randint(len(items), (len(items),), generator=rng).tolist()
        sub = pd.concat([df[df.item == items[i]] for i in pick])
        o = onsets(curves(sub))
        for lens in LENSES:
            for k, dist in boot_onsets[lens].items():
                if o[lens][k] is not None:
                    dist.append(o[lens][k])
        for k in gap_keys:
            gR, gJ = o["R"][k], o["J"][k]
            if gR is not None and gJ is not None:
                boot_diffs[k].append(gR - gJ)

    def ci_of(vals):
        if not len(vals):
            return None
        s = pd.Series(vals)
        return (float(s.quantile(0.025)), float(s.quantile(0.975)), len(vals) / max(n_boot, 1))

    def summary_of(vals):
        """Onsets are integers, so a percentile CI lands on mass points —
        report P(>0) and the histogram next to the interval."""
        if not len(vals):
            return None
        s = pd.Series(vals)
        return {"ci": (float(s.quantile(0.025)), float(s.quantile(0.975))),
                "p_gt_0": float((s > 0).mean()), "median": float(s.median()),
                "defined_frac": len(vals) / max(n_boot, 1),
                "hist": {float(k): int(v) for k, v in s.value_counts().sort_index().items()}}

    diffs = {k: ((full_onsets["R"][k] - full_onsets["J"][k])
                 if (full_onsets["R"][k] is not None and full_onsets["J"][k] is not None) else None)
             for k in gap_keys}
    return {
        "curves": full_curves, "onsets": full_onsets,
        "onset_cis": {lens: {k: ci_of(v) for k, v in dists.items()} for lens, dists in boot_onsets.items()},
        "gap_diffs": diffs,
        "gap_boot": {k: summary_of(v) for k, v in boot_diffs.items()},
        # back-compat for the existing report code
        "gap_R": full_onsets["R"].get("gap"), "gap_J": full_onsets["J"].get("gap"),
        "delta_R_minus_delta_J": diffs.get("gap"),
        "boot_ci": (ci_of(boot_diffs["gap"])[:2] if boot_diffs.get("gap") else None),
        "boot_defined_frac": (len(boot_diffs.get("gap", [])) / n_boot) if n_boot else 0.0,
    }
