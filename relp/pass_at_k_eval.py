"""pass@k lens-quality eval, reproducing the methodology behind
quantitative-evals/passk_qwen3.5-27b.md:

- 5 sets: multihop, multilingual, association, typo, poetry (order-ops excluded,
  matching the existing report).
- Readout position: last token of the tokenized prompt, EXCEPT poetry, where
  it's the token covering the prompt's last '\n' (end of couplet line 1).
- Correctness filter (multihop/multilingual only, since only they carry a
  `target`): keep an item iff the model's own greedy continuation of `prompt`
  contains `target` as a case-insensitive substring within the first 12 new
  tokens. association/typo/poetry have no `target` field, so nothing to
  filter against -- always kept, matching the existing report's near-100% keep
  rates for those three sets.
- Metric: for each (lens, set, layer), pass@k = mean over kept items of
  (# intermediates with lens-rank <= k at that layer) / (# intermediates).
  "mean pass@k" for a lens/set = the arithmetic mean of that per-layer curve
  over the given layer range (matches the existing report's summary numbers,
  which are lower than any single peak layer -- i.e. a curve-mean, not a
  best-layer or min-over-layers statistic).

One forward pass per item (hidden states at every layer) is reused across
every lens variant -- lenses only change the linear readout, not the forward
pass, and every rule-ablation lens has bit-identical forward values to the
unpatched model (verified separately), so no rule-patching is needed here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import torch
import transformers

sys.path.insert(0, "/workspace/results/lrp/jacobian-lens")
sys.path.insert(0, "/workspace/results/lrp")
import jlens

EVAL_DIR = "/workspace/results/lrp/jacobian-lens/data/evaluations"
SETS = ["multihop", "multilingual", "association", "typo", "poetry"]
K = 10


@dataclass
class EvalItem:
    set_name: str
    name: str
    prompt: str
    intermediates: list[str]
    target: str | None


def load_items(set_name: str) -> list[EvalItem]:
    with open(os.path.join(EVAL_DIR, f"lens-eval-{set_name}.json")) as f:
        raw = json.load(f)["items"]
    return [
        EvalItem(set_name, it["name"], it["prompt"], it["intermediates"], it.get("target"))
        for it in raw
    ]


def readout_position(item: EvalItem, tok, offsets) -> int:
    """Index into the prompt's tokenization to read the lens out at."""
    if item.set_name == "poetry":
        nl_char = item.prompt.rindex("\n")
        for i, (start, end) in enumerate(offsets):
            if start <= nl_char < end:
                return i
        # newline fell on a token boundary (end == nl_char): last token ending there
        for i in range(len(offsets) - 1, -1, -1):
            if offsets[i][1] == nl_char + 1:
                return i
        raise ValueError(f"couldn't locate newline token for {item.name!r}")
    return len(offsets) - 1


def single_token_id(tok, word: str) -> int | None:
    """First-token id for `word`, preferring the leading-space form (mid-sentence)."""
    for text in (" " + word, word):
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


@torch.no_grad()
def model_forward_hidden(model_lens, prompt: str, max_seq_len: int = 512):
    """Run one forward pass; return (input_ids[0], hidden_states dict {layer:[seq,d]}, offsets)."""
    from jlens.hooks import ActivationRecorder

    enc = model_lens.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_seq_len, return_offsets_mapping=True
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    input_ids = enc["input_ids"].to(model_lens.input_device)
    record_at = list(range(model_lens.n_layers))
    with ActivationRecorder(model_lens.layers, at=record_at) as rec:
        model_lens.forward(input_ids)
        hidden = {i: rec.activations[i][0].detach().float().cpu() for i in record_at}
    return input_ids[0].cpu(), hidden, offsets


@torch.no_grad()
def is_correct_continuation(hf_model, tok, prompt: str, target: str, device: str, max_new: int = 12) -> bool:
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = hf_model.generate(
        **inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id
    )
    continuation = tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return target.strip().lower() in continuation.lower()


def rank_of_token(logits_1d: torch.Tensor, token_id: int) -> int:
    """1-indexed rank of `token_id` in `logits_1d` (rank 1 = argmax)."""
    return int((logits_1d > logits_1d[token_id]).sum().item()) + 1


class LensVariant:
    """Wraps either a JacobianLens (transport + unembed) or plain logit-lens
    (unembed only) or a random-direction control, all reading from
    precomputed hidden states."""

    def __init__(self, name: str, kind: str, model_lens, lens: "jlens.JacobianLens | None" = None):
        self.name = name
        self.kind = kind  # "lens" | "logit" | "control"
        self.model_lens = model_lens
        self.lens = lens
        self._rand_dirs: dict[int, torch.Tensor] = {}

    def layers(self) -> list[int]:
        if self.kind == "lens":
            return self.lens.source_layers
        return list(range(self.model_lens.n_layers))

    def logits_at(self, hidden_at_layer: torch.Tensor, layer: int) -> torch.Tensor:
        h = hidden_at_layer.unsqueeze(0)
        if self.kind == "logit":
            v = h
        elif self.kind == "control":
            if layer not in self._rand_dirs:
                g = torch.Generator().manual_seed(1234 + layer)
                d = torch.randn(h.shape[-1], generator=g)
                self._rand_dirs[layer] = d / d.norm()
            v = (h.norm() * self._rand_dirs[layer]).unsqueeze(0)
        else:
            J = self.lens.jacobians[layer]
            v = h @ J.T
        return self.model_lens.unembed(v.to(self.model_lens.input_device)).float().cpu()[0]


def build_variants(model_lens, lens_paths: dict[str, str]) -> list[LensVariant]:
    variants = [LensVariant("logit", "logit", model_lens)]
    for name, path in lens_paths.items():
        lens = jlens.JacobianLens.load(path)
        variants.append(LensVariant(name, "lens", model_lens, lens))
    variants.append(LensVariant("control", "control", model_lens))
    return variants


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lenses", required=True, help="name=path,name=path,...")
    p.add_argument("--out_prefix", required=True)
    p.add_argument("--limit_per_set", type=int, default=None)
    args = p.parse_args()

    lens_paths = dict(kv.split("=", 1) for kv in args.lenses.split(","))

    device = "cuda:0"
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model_lens = jlens.from_hf(hf_model, tok)
    variants = build_variants(model_lens, lens_paths)
    print(f"[eval] variants: {[v.name for v in variants]}", flush=True)

    # results[set][variant][layer] -> list of per-item hit-fractions
    all_rows = []
    kept_counts = {}

    for set_name in SETS:
        items = load_items(set_name)
        if args.limit_per_set:
            items = items[: args.limit_per_set]
        kept_items = []
        for it in items:
            if it.target is not None:
                keep = is_correct_continuation(hf_model, tok, it.prompt, it.target, device)
            else:
                keep = True
            if keep:
                kept_items.append(it)
        kept_counts[set_name] = (len(kept_items), len(items))
        print(f"[eval] {set_name}: kept {len(kept_items)}/{len(items)}", flush=True)

        for it in kept_items:
            input_ids, hidden, offsets = model_forward_hidden(model_lens, it.prompt)
            pos = readout_position(it, tok, offsets)
            inter_ids = [tid for w in it.intermediates if (tid := single_token_id(tok, w)) is not None]
            if not inter_ids:
                continue
            for v in variants:
                for layer in v.layers():
                    logits = v.logits_at(hidden[layer][pos], layer)
                    hit_frac = sum(rank_of_token(logits, tid) <= K for tid in inter_ids) / len(inter_ids)
                    all_rows.append(
                        {"set": set_name, "variant": v.name, "layer": layer, "item": it.name, "hit_frac": hit_frac}
                    )

    import pandas as pd

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    df.to_parquet(f"{args.out_prefix}_raw.parquet")

    per_layer = df.groupby(["set", "variant", "layer"])["hit_frac"].mean().reset_index()
    per_layer.to_csv(f"{args.out_prefix}_per_layer.csv", index=False)

    summary = df.groupby(["set", "variant"])["hit_frac"].mean().reset_index()
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    with open(f"{args.out_prefix}_kept_counts.json", "w") as f:
        json.dump(kept_counts, f, indent=2)

    print(f"[eval] wrote {args.out_prefix}_{{raw.parquet,per_layer.csv,summary.csv}}", flush=True)
    print(summary.pivot(index="variant", columns="set", values="hit_frac"), flush=True)


if __name__ == "__main__":
    main()
