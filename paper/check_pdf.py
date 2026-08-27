#!/usr/bin/env python3
"""Post-build audit: read the RENDERED PDF and compare it to the frozen results.

A unit test on the plotting function proved nothing when `\\graphicspath` silently
resolved Figure 1 to a stale directory. The only artifact that matters is the
one a reader opens, so this reads that.

Usage:  python paper/check_pdf.py <paper.pdf> <statistical_results.json>
Exit 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

# Figure 1a prints these to two decimals; they must also appear in the tables.
PRIMARY_DIM = "contextual_coherence"


def pdf_text(path: Path) -> str:
    out = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                         capture_output=True, text=True, check=True)
    return out.stdout


def expected_means(stats: dict) -> dict:
    """Per-lens means, from either artifact shape.

    `analyse-v2` nests them at per_model[model]["means"][dimension]; the
    non-echo analysis writes a flat top-level "mean_scores". Accepting only one
    shape made the checker silently pass on a file it was not reading.
    """
    means = {}
    for model, entry in (stats.get("per_model") or {}).items():
        by_dim = (entry or {}).get("means") or {}
        per_lens = by_dim.get(PRIMARY_DIM) or by_dim.get(stats.get("dimension", "")) or {}
        if per_lens:
            means[model] = {l: round(float(v), 2) for l, v in per_lens.items()}
    if not means:
        for model, per_lens in (stats.get("mean_scores") or {}).items():
            means[model] = {l: round(float(v), 2) for l, v in per_lens.items()}
    return means


def main() -> int:
    pdf, stats_path = Path(sys.argv[1]), Path(sys.argv[2])
    text = pdf_text(pdf)
    stats = json.loads(stats_path.read_text())
    failures, checks = [], 0

    means = expected_means(stats)
    if not means:
        failures.append("statistical_results.json carries no per-lens means to check")
    for model, per_lens in means.items():
        for lens, value in per_lens.items():
            checks += 1
            if f"{value:.2f}" not in text:
                failures.append(
                    f"Figure 1a / tables: {model} {lens} mean {value:.2f} does not "
                    f"appear anywhere in the rendered PDF")

    # Means from a DIFFERENT scoring rule must be absent. Matched on a decimal
    # boundary: a bare substring test flags 0.78 inside the legitimate CI bound
    # 0.785, which is a false alarm, not a stale figure.
    stale_means = {"0.78", "1.42", "2.07", "2.39", "1.49"}
    for stale in sorted(stale_means - {f"{v:.2f}" for per in means.values()
                                       for v in per.values()}):
        checks += 1
        if re.search(rf"(?<![\d.]){re.escape(stale)}(?![\d])", text):
            failures.append(
                f"value {stale} appears in the PDF as a standalone number: this is "
                f"an adjudicated-rule mean and must not survive in a mean-of-two "
                f"document (this is exactly how the stale Figure 1 was missed)")

    # Headline contrasts only. Secondary dimensions are reported to 3dp in one
    # table; the primary three per model are what a reader quotes.
    for model, contrasts in (stats.get("per_model") or {}).items():
        for name in ("released-R - released-J", "released-R - logit",
                     "released-J - logit"):
            entry = contrasts.get(name)
            if not isinstance(entry, dict) or "delta" not in entry:
                continue
            checks += 1
            if f"{entry['delta']:.3f}" not in text:
                failures.append(f"{model} {name}: delta {entry['delta']:.3f} "
                                "not found in the rendered PDF")

    print(f"{checks - len(failures)}/{checks} PDF checks passed")
    for f in failures:
        print(f"  FAIL: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
