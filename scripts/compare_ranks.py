"""Diff two per-item rank parquets (C2 output) row for row.

Used as the C3 parity check: the same eval run with `--unembed-chunk 1` (the
pre-C3 one-readout-at-a-time path) and with the default chunk must produce
identical integer ranks. Batched and single-row GEMM kernels are not
bit-identical, so a handful of near-tie disagreements is acceptable; a
systematic shift is not.

    uv run python scripts/compare_ranks.py A.parquet B.parquet
"""

import sys
from pathlib import Path

import pandas as pd

KEYS = ["set", "item_id", "intermediate", "layer", "lens"]


def main(a_path: str, b_path: str) -> int:
    a = pd.read_parquet(a_path).set_index(KEYS)["rank"]
    b = pd.read_parquet(b_path).set_index(KEYS)["rank"]
    print(f"A {Path(a_path).name}: {len(a)} rows   B {Path(b_path).name}: {len(b)} rows")

    only_a, only_b = a.index.difference(b.index), b.index.difference(a.index)
    if len(only_a) or len(only_b):
        print(f"MISMATCHED KEYS: {len(only_a)} only in A, {len(only_b)} only in B")

    shared = a.index.intersection(b.index)
    a, b = a.loc[shared], b.loc[shared]
    diff = a != b
    n = int(diff.sum())
    print(f"shared rows: {len(shared)}   differing ranks: {n} ({n / max(len(shared), 1):.4%})")
    if n:
        delta = (b[diff] - a[diff])
        print(f"delta B-A: mean {delta.mean():+.3f}  median {delta.median():+.1f}  "
              f"min {delta.min():+d}  max {delta.max():+d}")
        print("\nworst 10 by |delta|:")
        print(pd.DataFrame({"A": a[diff], "B": b[diff], "delta": delta})
              .reindex(delta.abs().sort_values(ascending=False).index).head(10))
        # what actually matters: does pass@k flip?
        for k in (1, 5, 10, 50):
            flips = int(((a <= k) != (b <= k)).sum())
            print(f"pass@{k:<3} decisions flipped: {flips} ({flips / len(shared):.4%})")
    return 1 if len(only_a) or len(only_b) else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
