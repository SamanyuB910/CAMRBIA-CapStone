"""R-lens replication: LRP-rule patching + fitting harness around the official jlens.

Modules: rules (the LRP rules + patcher), fit (patch -> jlens.fit -> save),
analysis (readout tables + verification metrics), evals (the pass@10 battery),
stats (C5), figures (C6), cli (the `rlens` command).

``rules``/``fit``/``evals`` need torch; ``stats`` and ``figures`` need only
pandas/numpy/matplotlib and read the committed parquets. Importing the package
must therefore NOT pull torch in eagerly, or the whole analysis layer becomes
un-runnable on a machine without it (e.g. a laptop doing plots while the GPU
box runs the evals). Hence the lazy re-export below.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers / IDEs only - never executed at runtime
    from rlens.rules import RulesConfig, RulesPatcher, apply_rules

__all__ = ["RulesConfig", "RulesPatcher", "apply_rules"]

_LAZY = {"RulesConfig": "rlens.rules", "RulesPatcher": "rlens.rules", "apply_rules": "rlens.rules"}


def __getattr__(name: str):
    """PEP 562 lazy attribute access: ``rlens.RulesConfig`` still works, but the
    torch import only happens when something actually asks for it."""
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
