"""The control lens: a random, norm-matched dense transport.

Without it, "R-lens beats J-lens" is ambiguous — a dense matrix multiply moves a
residual out of the early-layer basis and toward *something*, and the logit lens
(identity) is not a fair floor for that. The control answers the narrower
question the post's numbers actually rest on: is R-lens beating J-lens, or is
**any** dense transport of the right magnitude better than the identity?

Construction: per source layer, an iid Gaussian matrix rescaled so
``‖J_ctrl‖_F == ‖J_R‖_F`` exactly. Everything is derived from one integer seed
(``pins.yaml`` ``fitting.seed``) plus the layer index, so the control is
reproducible without shipping another multi-GB file. CUDA and CPU RNG streams
differ, so a control run is reproducible *on the same device kind* — record
which one produced a given results file.

Matrices are regenerated per call rather than stored: a 27B layer is
``5120² × fp32 = 105 MB``, so materialising all ~60 would cost ~6.3 GB of VRAM
next to a 56 GB model and two ~6 GB lenses. ``torch.randn`` on an A100 is far
cheaper than that headroom. Small lenses (4b) fall under ``cache_bytes`` and are
kept instead.
"""

from __future__ import annotations

import torch

CACHE_BYTES = 2 * 1024**3  # cache the whole control lens below this size


class ControlLens:
    """Duck-typed stand-in for ``jlens.JacobianLens``: exposes ``source_layers``
    and ``transport(residual, layer)``, which is all ``run_passk`` needs."""

    def __init__(self, reference, *, seed: int, cache_bytes: int = CACHE_BYTES) -> None:
        """Args:
            reference: the lens whose per-layer Frobenius norms are matched
                (use the R-lens — it is the arm the control is the null for).
            seed: base seed; layer ``l`` uses ``seed + l``.
        """
        self.source_layers = list(reference.source_layers)
        self.d_model = int(reference.d_model)
        self.seed = int(seed)
        self.frob = {
            layer: float(reference.jacobians[layer].float().norm())
            for layer in self.source_layers
        }
        size = self.d_model**2 * 4 * len(self.source_layers)
        self._cache: dict[int, torch.Tensor] | None = {} if size <= cache_bytes else None

    def __repr__(self) -> str:
        held = "cached" if self._cache is not None else "regenerated per call"
        return (
            f"ControlLens(d_model={self.d_model}, seed={self.seed}, "
            f"source_layers=[{self.source_layers[0]}..{self.source_layers[-1]}] "
            f"({len(self.source_layers)} layers), {held})"
        )

    def matrix(self, layer: int, device, dtype=torch.float32) -> torch.Tensor:
        """The control matrix for ``layer``: iid Gaussian, rescaled to the
        reference lens's Frobenius norm. Deterministic in ``(seed, layer)``."""
        if self._cache is not None and layer in self._cache:
            J = self._cache[layer]
            if J.device == torch.device(device) and J.dtype == dtype:
                return J
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed + layer)
        J = torch.randn(self.d_model, self.d_model, generator=gen, device=device, dtype=dtype)
        J *= self.frob[layer] / float(J.float().norm())
        if self._cache is not None:
            self._cache[layer] = J
        return J

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """``J_ctrl @ h`` — same signature as ``JacobianLens.transport``."""
        return residual @ self.matrix(layer, residual.device, residual.dtype).T
