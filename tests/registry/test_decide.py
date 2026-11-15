from nsh_pipeline.registry.decide import (
    Check, candidate_metrics, decide, gate_checks, registry_settings,
)

VALIDATION = {"min_pr_auc": 0.20, "min_pr_auc_lift": 1.5, "min_roc_auc": 0.65, "min_recall": 0.40}
PROD = {"version": "v1"}


def bias(pr_auc=0.32, passed=True, selected="lr", recall=0.68):
    return {"selected_model": selected, "passed": passed,
            "performance": {"test": {"pr_auc": pr_auc, "roc_auc": 0.7, "recall": recall, "precision": 0.25,
                                     "f1": 0.36, "positive_rate": 0.13}}}


def by_name(checks):
    return {c.name: c for c in checks}


def test_settings_fill_defaults_and_merge_gcp():
    s = registry_settings({"registry": {"metric": "roc_auc", "gcp": {"location": "us-central1"}}})
    assert s["metric"] == "roc_auc" and s["backend"] == "local"
    assert s["gcp"] == {"location": "us-central1", "repository": "nsh-models", "package": "nsh-denial-model"}


def test_candidate_metrics_adds_lift():
    assert round(candidate_metrics(bias(pr_auc=0.26))["pr_auc_lift"], 2) == 2.0


def test_all_gates_pass():
    checks = by_name(gate_checks({"passed": True}, bias(), VALIDATION))
    assert all(c.passed for c in checks.values())


def test_validation_failure_and_missing_reports():
    assert not by_name(gate_checks({"passed": False}, bias(), VALIDATION))["validation_gate"].passed
    checks = by_name(gate_checks(None, None, VALIDATION))
    assert not checks["validation_gate"].passed and not checks["bias_gate"].passed


def test_bias_gate_needs_a_selected_model():
    assert not by_name(gate_checks({"passed": True}, bias(passed=True, selected=None), VALIDATION))["bias_gate"].passed
    assert not by_name(gate_checks({"passed": True}, bias(passed=False), VALIDATION))["bias_gate"].passed


def test_selected_model_held_to_validation_minimums():
    check = by_name(gate_checks({"passed": True}, bias(recall=0.30), VALIDATION))["candidate_metrics"]
    assert not check.passed and "recall" in check.detail


def test_missing_test_metrics_fail_candidate_check():
    report = {"selected_model": "lr", "passed": True, "performance": {}}
    assert not by_name(gate_checks({"passed": True}, report, VALIDATION))["candidate_metrics"].passed


def test_first_version_is_promoted():
    d = decide([Check("g", True, "ok")], 0.3, None, None, None)
    assert d.promote and "first version" in d.reason


def test_better_candidate_promoted_worse_kept():
    assert decide([], 0.33, PROD, 0.30, "rescored").promote
    worse = decide([], 0.29, PROD, 0.30, "rescored")
    assert not worse.promote and worse.reason.startswith("rolled back: kept production v1")


def test_equal_or_within_min_improvement_is_kept():
    assert not decide([], 0.30, PROD, 0.30, "rescored").promote
    assert not decide([], 0.305, PROD, 0.30, "rescored", min_improvement=0.01).promote


def test_failing_gate_never_promoted_even_if_better():
    d = decide([Check("bias_gate", False, "unfair")], 0.9, PROD, 0.1, "recorded")
    assert not d.promote and "unfair" in d.reason


def test_missing_production_metric_blocks():
    assert not decide([], 0.3, PROD, None, None).promote


def test_failing_without_production():
    d = decide([Check("validation_gate", False, "bad")], 0.3, None, None, None)
    assert not d.promote and d.reason.startswith("not promoted")
    assert d.to_dict()["checks"][0]["name"] == "validation_gate"


def test_float_noise_is_not_an_improvement():
    assert not decide([], 0.32480001, PROD, 0.3248, "rescored").promote
    assert decide([], 0.3249, PROD, 0.3248, "rescored").promote
