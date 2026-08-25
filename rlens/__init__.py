"""R-lens replication: LRP-rule patching + fitting harness around the official jlens.

Modules: rules (the LRP rules + patcher), fit (patch -> jlens.fit -> save),
analysis (readout tables + verification metrics), cli (the `rlens` command).
"""

from rlens.rules import RulesConfig, RulesPatcher, apply_rules

__all__ = ["RulesConfig", "RulesPatcher", "apply_rules"]
