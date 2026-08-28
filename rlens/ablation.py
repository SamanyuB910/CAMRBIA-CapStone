"""Core experiment 3: band ablation of lens directions, scored by accuracy loss.

The R-lens post's protocol, followed as closely as we can:

    "on 30 questions from the multihop eval set, we ablate the projection of the
     R-lens, J-lens, and logit lens directions ... for the intermediate token
     from the activations", "ablating on the penultimate token position of the
     prompt", over "the first half of layers (to test saliency of early
     representations)" and "all layers". "We test the relative accuracy loss
     (judged with a GPT 5.4 nano autorater): we sample each prompt 8 times
     before and after ablating, and measure the reduction in accuracy".

Two deliberate deviations, both reported in the output:
- **grader**: a deterministic normalized matcher instead of an LLM autorater (no
  API key here). Multihop targets are short factual words, and every per-sample
  decision is written to the records so disputed items can be checked by hand.
- **controls**: we add norm-matched random-direction ablations, which the post
  does not have. Without them "ablation hurts" cannot be separated from "any
  displacement of this size hurts".

Everything else -- what is ablated, where, over which layer bands, how many
samples -- follows the quoted protocol.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from rlens.evals import _synonyms
from rlens.onset import _layer_jacobians, _stable_seed, lens_vectors, make_ablation

REPO_ROOT = Path(__file__).resolve().parents[1]
LENSES = ("R", "J", "logit")
N_SAMPLES = 8          # the post samples each prompt 8 times
MAX_NEW_TOKENS = 12
TEMPERATURE = 1.0      # not specified in the post; documented as our choice
N_RANDOM = 3
MIN_CLEAN_ACC = 0.5    # "filter to questions the model answers correctly"


@dataclass
class AblationItem:
    name: str
    prompt: str            # rstripped
    target: str            # the final answer, what accuracy is scored on
    intermediate: str      # the concept whose lens direction is ablated
    intermediate_id: int   # single-token id for the direction


def bands(n_layers: int) -> dict[str, list[int]]:
    """The post's two arms: the first half of layers, and all layers."""
    return {"first_half": list(range(n_layers // 2)), "all_layers": list(range(n_layers))}


def graded_correct(continuation: str, target: str) -> bool:
    """Deterministic stand-in for the autorater.

    True when any surface form of the target (case variants, and digit<->word
    for numeric answers, via the eval battery's synonym logic) appears as a
    whole word in the sampled continuation.
    """
    text = continuation.lower()
    for form in _synonyms(target):
        if re.search(rf"\b{re.escape(form.lower())}\b", text):
            return True
    return False


def build_items(tok, *, limit: int | None = None) -> tuple[list[AblationItem], list[dict]]:
    """Multihop eval items with exactly one single-token intermediate."""
    from rlens.onset import _direction_id

    raw = json.loads(
        (REPO_ROOT / "data" / "eval_prompts" / "evaluations" / "lens-eval-multihop.json").read_text(encoding="utf-8")
    )["items"][:limit]
    kept, log = [], []
    for item in raw:
        inters = item.get("intermediates", [])
        if len(inters) != 1:
            log.append({"name": item["name"], "kept": False, "reason": f"{len(inters)} intermediates"}); continue
        tid = _direction_id(tok, inters[0])
        if tid is None:
            log.append({"name": item["name"], "kept": False, "reason": "multi-token intermediate"}); continue
        kept.append(AblationItem(name=item["name"], prompt=item["prompt"].rstrip(),
                                 target=item["target"], intermediate=inters[0], intermediate_id=tid))
        log.append({"name": item["name"], "kept": True})
    return kept, log


class BandAblator:
    """Applies an edit at ONE prompt position across a band of layers, during
    generation.

    With a KV cache the block sees the whole prompt on the prefill pass and a
    single token on every decode step. The edit belongs to the prompt, so it
    must fire only while ``seq_len > 1``; editing on decode steps would corrupt
    positions the protocol never touches.
    """

    def __init__(self, hf_model):
        self.layers = hf_model.model.layers

    def __call__(self, edits_by_layer: dict[int, object], position: int):
        handles = []

        def make_hook(fn):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                if h.shape[1] > 1:  # prefill only
                    h[:, position] = fn(h[:, position].float()).to(h.dtype)
                return output

            return hook

        class _ctx:
            def __enter__(_):
                for layer, fn in edits_by_layer.items():
                    handles.append(self.layers[layer].register_forward_hook(make_hook(fn)))
                return _

            def __exit__(_, *exc):
                for handle in handles:
                    handle.remove()

        return _ctx()


@torch.no_grad()
def sample_answers(hf_model, tok, prompt: str, *, n: int, seed: int,
                   max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = TEMPERATURE) -> list[str]:
    """n sampled continuations, batched, reproducible for a given seed."""
    device = hf_model.model.embed_tokens.weight.device
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    torch.manual_seed(seed)
    out = hf_model.generate(
        input_ids.expand(n, -1), max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=1.0, pad_token_id=tok.eos_token_id,
    )
    return [tok.decode(row[input_ids.shape[1]:], skip_special_tokens=True) for row in out]


@torch.no_grad()
def run_ablation_study(hf_model, tok, lenses: dict, items: list[AblationItem], *,
                       n_samples: int = N_SAMPLES, temperature: float = TEMPERATURE) -> pd.DataFrame:
    """Clean + {lens x band} + random-control conditions, 8 samples each."""
    n_layers = hf_model.config.get_text_config().num_hidden_layers
    arms = bands(n_layers)
    ablator = BandAblator(hf_model)
    records = []

    for item in items:
        seq_len = tok(item.prompt, return_tensors="pt").input_ids.shape[1]
        position = seq_len - 2  # the post ablates the PENULTIMATE prompt token

        def emit(condition: str, arm: str, texts: list[str]) -> None:
            for i, text in enumerate(texts):
                records.append({"item": item.name, "condition": condition, "arm": arm, "sample": i,
                                "target": item.target, "intermediate": item.intermediate,
                                "text": text, "correct": int(graded_correct(text, item.target))})

        emit("clean", "-", sample_answers(hf_model, tok, item.prompt, n=n_samples,
                                          seed=_stable_seed(item.name, "clean"), temperature=temperature))

        # per-layer directions for the intermediate, one lens at a time
        dirs = {lens: {} for lens in LENSES}
        for l in range(n_layers):
            per_layer = _layer_jacobians(lenses, l)
            for lens in LENSES:
                dirs[lens][l] = lens_vectors(hf_model, per_layer[lens], [item.intermediate_id])[0]

        for arm, layer_set in arms.items():
            for lens in LENSES:
                edits = {l: make_ablation(dirs[lens][l], 1.0) for l in layer_set}
                with ablator(edits, position):
                    texts = sample_answers(hf_model, tok, item.prompt, n=n_samples,
                                           seed=_stable_seed(item.name, lens, arm), temperature=temperature)
                emit(lens, arm, texts)

            # norm-matched random controls (not in the post; separates
            # "this direction matters" from "any displacement of this size does")
            for b in range(N_RANDOM):
                edits = {}
                for l in layer_set:
                    v = dirs["R"][l]
                    g = torch.randn(v.shape[0], generator=torch.Generator().manual_seed(
                        _stable_seed(item.name, l, arm, b)))
                    edits[l] = make_ablation(g / g.norm() * v.norm(), 1.0)
                with ablator(edits, position):
                    texts = sample_answers(hf_model, tok, item.prompt, n=n_samples,
                                           seed=_stable_seed(item.name, "random", arm, b), temperature=temperature)
                emit("random", arm, texts)
    return pd.DataFrame(records)


def analyze(df: pd.DataFrame, *, min_clean_acc: float = MIN_CLEAN_ACC,
            n_boot: int = 1000, seed: int = 0) -> dict:
    """Relative accuracy loss per (lens, arm), bootstrapped over items."""
    acc = df.groupby(["item", "condition", "arm"])["correct"].mean().reset_index()
    clean = acc[acc.condition == "clean"].set_index("item")["correct"]
    keep = clean[clean >= min_clean_acc].index
    rng = torch.Generator().manual_seed(seed)

    rows, boot = {}, {}
    for cond in (*LENSES, "random"):
        for arm in ("first_half", "all_layers"):
            sub = acc[(acc.condition == cond) & (acc.arm == arm) & (acc.item.isin(keep))]
            sub = sub.groupby("item")["correct"].mean()  # averages the 3 random seeds
            if not len(sub):
                continue
            items = sorted(sub.index)
            base = clean.loc[items]
            loss = ((base - sub.loc[items]) / base)
            rows[(cond, arm)] = {"clean_acc": float(base.mean()), "ablated_acc": float(sub.loc[items].mean()),
                                 "rel_acc_loss": float(loss.mean()), "n": len(items)}
            draws = [float(loss.iloc[torch.randint(len(loss), (len(loss),), generator=rng).tolist()].mean())
                     for _ in range(n_boot)]
            s = pd.Series(draws)
            boot[(cond, arm)] = (float(s.quantile(0.025)), float(s.quantile(0.975)))
    table = pd.DataFrame(rows).T
    table["ci_low"] = [boot[k][0] for k in table.index]
    table["ci_high"] = [boot[k][1] for k in table.index]

    # primary comparison: R vs J, per arm, paired over items
    contrasts = {}
    for arm in ("first_half", "all_layers"):
        per = {}
        for cond in ("R", "J"):
            sub = acc[(acc.condition == cond) & (acc.arm == arm) & (acc.item.isin(keep))].set_index("item")["correct"]
            per[cond] = ((clean.loc[sorted(sub.index)] - sub.loc[sorted(sub.index)]) / clean.loc[sorted(sub.index)])
        common = sorted(set(per["R"].index) & set(per["J"].index))
        if not common:
            continue
        diff = per["R"].loc[common] - per["J"].loc[common]
        draws = [float(diff.iloc[torch.randint(len(diff), (len(diff),), generator=rng).tolist()].mean())
                 for _ in range(n_boot)]
        s = pd.Series(draws)
        contrasts[arm] = {"R_minus_J": float(diff.mean()),
                          "ci": (float(s.quantile(0.025)), float(s.quantile(0.975))),
                          "p_R_gt_J": float((s > 0).mean())}
    return {"table": table, "contrasts": contrasts, "n_items": len(keep),
            "n_dropped_by_filter": int(len(clean) - len(keep))}


def save_items(items: list[AblationItem], log: list[dict], path: Path) -> None:
    path.write_text(json.dumps({"items": [asdict(i) for i in items], "filter_log": log}, indent=1),
                    encoding="utf-8")
