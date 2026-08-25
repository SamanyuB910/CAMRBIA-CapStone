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

import json
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
        A = Gd + ridge * torch.eye(2, device=h.device, dtype=torch.float64)
        z = torch.linalg.solve(A, rhs) if ridge > 0 else torch.linalg.lstsq(A, rhs).solution
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
    """Run one prompt under many single-(layer, position) edits in one batch.

    Post-forward hook on decoder block ``layer`` (same tensor the lens was fit
    against); batch element b gets ``edits[b]`` applied to h[b, position] in
    fp32. Returns final-position logits [n_edits, vocab] (fp32, CPU).
    """

    def __init__(self, hf_model):
        self.hf_model = hf_model
        self.layers = hf_model.model.layers

    @torch.no_grad()
    def run(self, input_ids: torch.Tensor, layer: int, position: int, edits: list, batch_size: int = 24) -> torch.Tensor:
        out = []
        for start in range(0, len(edits), batch_size):
            chunk = edits[start : start + batch_size]
            batch = input_ids.expand(len(chunk), -1)

            def hook(module, args, output, chunk=chunk):
                h = output[0] if isinstance(output, tuple) else output
                for b, edit in enumerate(chunk):
                    h[b, position] = edit(h[b, position].float()).to(h.dtype)
                return output

            handle = self.layers[layer].register_forward_hook(hook)
            try:
                logits = self.hf_model(input_ids=batch, use_cache=False).logits
            finally:
                handle.remove()
            out.append(logits[:, -1].float().cpu())
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
    y_prime_id: int
    t_i: int               # absolute intervention/readout position


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
        if c.lower() in prompt.lower() or c_prime.lower() in prompt.lower():
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
        control_ok, _, _ = top1_and_logp(control, yp_set)
        if not control_ok:
            drop(item, "control answer wrong"); continue

        cat_items = by_cat[item["category"]]
        nxt = cat_items[(cat_items.index(item) + 1) % len(cat_items)]
        c_wrong = next(
            (w for w in (nxt["swap_to"], nxt["intermediate"]) if w.lower() not in (c.lower(), c_prime.lower()) and _direction_id(tok, w)),
            None,
        )
        if c_wrong is None:
            drop(item, "no wrong-concept candidate"); continue

        seq_len = tok(prompt, return_tensors="pt").input_ids.shape[1]
        y_id = y_set[int(logp[y_set].argmax())]
        yp_id = yp_set[int(logp[yp_set].argmax())]
        kept.append(
            OnsetItem(
                name=item["name"], category=item["category"], prompt=prompt,
                control_prompt=control, c=c, c_prime=c_prime, c_wrong=c_wrong,
                y=y, y_prime=y_prime,
                c_id=ids["c"], c_prime_id=ids["c'"], c_wrong_id=_direction_id(tok, c_wrong),
                y_id=y_id, y_prime_id=yp_id,
                t_i=seq_len + position if position < 0 else position,
            )
        )
        log.append({"name": item["name"], "kept": True})
    return kept, log


# ---------------------------------------------------------------------------
# Measurements (§7.3-7.5, §7.7)
# ---------------------------------------------------------------------------


def _layer_jacobians(lenses: dict, layer: int) -> dict[str, torch.Tensor | None]:
    """Per-layer J for each lens name; the logit lens has no transport."""
    return {"R": lenses["R"].jacobians[layer], "J": lenses["J"].jacobians[layer], "logit": None}


def _edit_battery(hf_model, jacobians: dict[str, torch.Tensor | None], item: OnsetItem, *, ridge: float, center_mu: torch.Tensor | None):
    """All (lens, condition, alpha) edits for one (item, layer).
    Returns (labels, edits) where labels are (lens, condition, alpha) tuples."""
    labels, edits = [], []

    def add(lens, condition, alpha, edit):
        labels.append({"lens": lens, "condition": condition, "alpha": alpha})
        edits.append(edit)

    add("-", "identity", 0.0, lambda h: h)  # alpha=0 numerics check
    for lens in LENSES:
        J = jacobians[lens]
        v_c, v_cp, v_wrong, v_y, v_yp = lens_vectors(
            hf_model, J, [item.c_id, item.c_prime_id, item.c_wrong_id, item.y_id, item.y_prime_id]
        )
        add(lens, "ablate", 1.0, make_ablation(v_c, 1.0, mu=center_mu))
        add(lens, "ablate_wrong", 1.0, make_ablation(v_wrong, 1.0, mu=center_mu))
        add(lens, "ablate_answer", 1.0, make_ablation(v_y, 1.0, mu=center_mu))
        for alpha in ALPHAS:
            add(lens, "swap", alpha, make_pinv_swap(v_c, v_cp, alpha, ridge=ridge))
        add(lens, "swap_reflection", 1.0, make_reflection_swap(v_c, v_cp, 1.0))
        add(lens, "swap_wrong", 1.0, make_pinv_swap(v_wrong, v_cp, 1.0, ridge=ridge))
        add(lens, "swap_answer", 1.0, make_pinv_swap(v_y, v_yp, 1.0, ridge=ridge))
    return labels, edits


@torch.no_grad()
def run_measurements(
    hf_model, tok, lenses: dict[str, "object"], items: list[OnsetItem], *,
    layers: list[int], ridge: float = 0.0, center: bool = False, batch_size: int = 24,
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
                    h = rec.activations[l][0, item.t_i].float().cpu()
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
        base = {"item": item.name, "category": item.category, "clean_logp_y": float(clean_logp[item.y_id]), "clean_margin": clean_margin}

        # shuffled-label baseline concept for the readout (fixed derangement)
        shuf = items[(item_idx + 1) % len(items)]

        for layer in layers:
            h_t = acts[layer][item.t_i].float().cpu()
            jacobians = _layer_jacobians(lenses, layer)
            for lens in LENSES:
                J = jacobians[lens]
                read = h_t if J is None else h_t @ J.float().T
                logits_l = unembed(read)
                for concept, cond in ((item.c, "readout"), (shuf.c, "readout_shuffled")):
                    ids = token_ids_of(tok, concept)
                    records.append({**base, "layer": layer, "lens": lens, "condition": cond, "alpha": 0.0, "rank_c": rank_of(logits_l, ids)})

            labels, edits = _edit_battery(hf_model, jacobians, item, ridge=ridge, center_mu=mu_by_layer.get(layer))
            # norm-matched random controls for the real ablation and swap(1) of each lens
            deltas = {}
            for label, edit in zip(labels, edits):
                if label["condition"] in ("ablate", "swap") and label["alpha"] == 1.0:
                    deltas[(label["lens"], label["condition"])] = float((edit(h_t.clone()) - h_t).norm())
            for (lens, cond), dn in deltas.items():
                for b in range(N_RANDOM):
                    labels.append({"lens": lens, "condition": f"random_{cond}", "alpha": 1.0})
                    edits.append(make_random_displacement(dn, h_t.shape[0], seed=hash((item.name, layer, lens, cond, b)) & 0x7FFFFFFF))

            logits = runner.run(input_ids, layer, item.t_i, edits, batch_size=batch_size)
            logp = logits.log_softmax(-1)
            for label, lp, lg in zip(labels, logp, logits):
                records.append({
                    **base, "layer": layer, **label,
                    "logp_y": float(lp[item.y_id]), "logp_yp": float(lp[item.y_prime_id]),
                    "margin": float(lp[item.y_prime_id] - lp[item.y_id]),
                    "top1_is_yp": int(int(lg.argmax()) in token_ids_of(tok, item.y_prime)),
                    "top1_is_y": int(int(lg.argmax()) in token_ids_of(tok, item.y)),
                })
        # per-item swap diagnostics at a middle layer for the report
        mid = layers[len(layers) // 2]
        for lens, J in _layer_jacobians(lenses, mid).items():
            v_c, v_cp = lens_vectors(hf_model, J, [item.c_id, item.c_prime_id])
            records.append({**base, "layer": mid, "lens": lens, "condition": "diagnostics", "alpha": 0.0, **swap_diagnostics(v_c, v_cp)})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Analysis (§7.8-7.9)
# ---------------------------------------------------------------------------


def _onset(curve: pd.Series, rho: float, w: int = 2) -> float | None:
    excess = curve.to_numpy()
    peak = excess.max()
    if peak <= 0:
        return None
    norm = excess / (peak + 1e-9)
    layers = curve.index.to_numpy()
    for i in range(len(norm) - w + 1):
        if (norm[i : i + w] >= rho).all():
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

            out[lens] = {
                "R": by_layer("readout", "hit10") - by_layer("readout_shuffled", "hit10"),
                "N": by_layer("ablate", "necessity", 1.0) - by_layer("random_ablate", "necessity"),
                "S": by_layer("swap", "swap_effect", 1.0) - by_layer("random_swap", "swap_effect"),
            }
        return out

    def onsets(cv: dict) -> dict:
        res = {}
        for lens in LENSES:
            L_R = _onset(cv[lens]["R"], rho, w)
            L_N = _onset(cv[lens]["N"], rho, w)
            L_S = _onset(cv[lens]["S"], rho, w)
            joint = None
            if L_N is not None and L_S is not None:
                nn = cv[lens]["N"] / (cv[lens]["N"].max() + 1e-9)
                ss = cv[lens]["S"] / (cv[lens]["S"].max() + 1e-9)
                both = pd.Series(((nn >= rho) & (ss >= rho)).to_numpy().astype(float), index=nn.index)
                for i in range(len(both) - w + 1):
                    if both.iloc[i : i + w].all():
                        joint = float(both.index[i]); break
            res[lens] = {"L_R": L_R, "L_N": L_N, "L_S": L_S, "L_causal": joint,
                         "gap": (joint - L_R) if (joint is not None and L_R is not None) else None}
        return res

    full_curves = curves(df)
    full_onsets = onsets(full_curves)

    boot_gaps = []
    for _ in range(n_boot):
        pick = torch.randint(len(items), (len(items),), generator=rng).tolist()
        chosen = [items[i] for i in pick]
        sub = pd.concat([df[df.item == it] for it in chosen])
        o = onsets(curves(sub))
        gR, gJ = o["R"]["gap"], o["J"]["gap"]
        if gR is not None and gJ is not None:
            boot_gaps.append(gR - gJ)
    gap_diff = full_onsets["R"]["gap"], full_onsets["J"]["gap"]
    ci = (float(pd.Series(boot_gaps).quantile(0.025)), float(pd.Series(boot_gaps).quantile(0.975))) if boot_gaps else None
    return {"curves": full_curves, "onsets": full_onsets, "gap_R": gap_diff[0], "gap_J": gap_diff[1],
            "delta_R_minus_delta_J": (gap_diff[0] - gap_diff[1]) if None not in gap_diff else None,
            "boot_ci": ci, "boot_defined_frac": len(boot_gaps) / n_boot}
