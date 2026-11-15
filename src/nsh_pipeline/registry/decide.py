"""Promotion decision: may the newly selected model replace the production model?

The candidate is the model the bias stage selected (bias.json -> selected_model_dir). That can be a
mitigated model in models_mitigated/, not models/best, which is why the registry reads bias.json and not
the training output.

A candidate is promoted only when ALL checks pass:
  1. validation_gate   reports/model/validation.json -> passed (Raj's hold-out gate on models/best)
  2. bias_gate         reports/model/bias.json -> passed, and a model was selected (Het's fairness gate)
  3. candidate_metrics the selected model's own test metrics meet the validation minimums. Needed because
                       the bias stage may select a different model than the one validation.json describes.
  4. beats_production  candidate metric > production metric + min_improvement, both measured on the SAME
                       test split (production is re-scored when its model is available). No production
                       version yet: passes ("first version").
Otherwise production is kept (a rollback) and the reason is recorded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from nsh_pipeline.modeling.evaluate import apply_gate

METRIC_DECIMALS = 4   # bias.json / validation.json precision

DEFAULTS = {
    "backend": "local",                       # local | gcp; REGISTRY_BACKEND overrides
    "local_dir": "registry",
    "production_file": "registry/production.json",
    "decision_file": "reports/model/registry_decision.json",
    "metric": "pr_auc",
    "min_improvement": 0.0,
    "rescore_production": True,
    "alerts": True,
    "gcp": {"location": "us-east1", "repository": "nsh-models", "package": "nsh-denial-model"},
}


def registry_settings(params: dict) -> dict:
    """params.yaml -> registry with defaults filled in (gcp merged key by key)."""
    given = dict(params.get("registry") or {})
    settings = {**DEFAULTS, **given}
    settings["gcp"] = {**DEFAULTS["gcp"], **(given.get("gcp") or {})}
    return settings


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class Decision:
    promote: bool
    reason: str
    checks: list[Check] = field(default_factory=list)
    candidate_metric: float | None = None
    production_metric: float | None = None
    production_metric_source: str | None = None   # "rescored" | "recorded" | None (no production)

    def to_dict(self) -> dict:
        return asdict(self)


def candidate_metrics(bias_report: dict) -> dict:
    """Test-split metrics of the selected model (bias.json -> performance.test) plus PR-AUC lift."""
    test = dict((bias_report.get("performance") or {}).get("test") or {})
    if test.get("pr_auc") is not None and test.get("positive_rate"):
        test["pr_auc_lift"] = test["pr_auc"] / test["positive_rate"]
    return test


def gate_checks(validation_report: dict | None, bias_report: dict | None, validation_settings: dict) -> list[Check]:
    """Checks 1-3: the upstream gates, and the selected model against the validation minimums."""
    checks = []
    if validation_report is None:
        checks.append(Check("validation_gate", False, "reports/model/validation.json not found"))
    else:
        passed = bool(validation_report.get("passed"))
        checks.append(Check("validation_gate", passed,
                            "validation gate passed" if passed else "validation gate failed for models/best"))

    if bias_report is None:
        checks.append(Check("bias_gate", False, "reports/model/bias.json not found"))
        return checks
    selected = bias_report.get("selected_model")
    passed = bool(bias_report.get("passed")) and bool(selected)
    checks.append(Check("bias_gate", passed,
                        f"fairness gate passed for {selected}" if passed else "no candidate passed the fairness gate"))

    metrics = candidate_metrics(bias_report)
    try:
        results = apply_gate(metrics, validation_settings)
    except (KeyError, TypeError) as exc:
        checks.append(Check("candidate_metrics", False, f"bias.json has no test metric {exc} for the selected model"))
        return checks
    failed = [r for r in results if not r["passed"]]
    detail = ("selected model meets every validation minimum on test" if not failed else
              "; ".join(f"{r['metric']} {r['value']:.4f} < {r['minimum']:.4f}" for r in failed))
    checks.append(Check("candidate_metrics", not failed, detail))
    return checks


def decide(checks: list[Check], candidate_metric: float | None, production: dict | None,
           production_metric: float | None, production_metric_source: str | None,
           metric: str = "pr_auc", min_improvement: float = 0.0) -> Decision:
    """Combine the gate checks with the comparison against production (check 4)."""
    checks = list(checks)
    if production is None:
        checks.append(Check("beats_production", True, "no production model yet: first version"))
    elif candidate_metric is None or production_metric is None:
        checks.append(Check("beats_production", False, f"cannot compare {metric}: a value is missing"))
    else:
        # Compared at the reports' precision, so float noise never counts as an improvement.
        better = round(candidate_metric - production_metric - min_improvement, METRIC_DECIMALS) > 0
        sign = ">" if better else "<="
        checks.append(Check(
            "beats_production", better,
            f"candidate {metric} {candidate_metric:.4f} {sign} production {production['version']} "
            f"{production_metric:.4f} ({production_metric_source}) + {min_improvement}"))

    failed = [c for c in checks if not c.passed]
    if not failed:
        reason = "promoted: " + checks[-1].detail
    elif production is None:
        reason = "not promoted (no production model to keep): " + "; ".join(c.detail for c in failed)
    else:
        reason = f"rolled back: kept production {production['version']}: " + "; ".join(c.detail for c in failed)
    return Decision(promote=not failed, reason=reason, checks=checks, candidate_metric=candidate_metric,
                    production_metric=production_metric, production_metric_source=production_metric_source)
