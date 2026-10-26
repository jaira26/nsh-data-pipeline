"""params.injected.yaml reproduces the detection evidence; it must not drift from params.yaml."""
from nsh_pipeline.config import load_params
from tests.validation.conftest import ROOT

# The only values that may differ, and what they must be in the injected file.
ALLOWED = {("inject_issues", "enabled"): True, ("validation", "mode"): "report"}


def flatten(value, prefix=()):
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            out.update(flatten(child, prefix + (key,)))
        return out
    return {prefix: value}


def test_injected_params_differ_only_in_injection_and_mode():
    base = flatten(load_params(ROOT / "params.yaml"))
    injected = flatten(load_params(ROOT / "params.injected.yaml"))
    differences = {key for key in base.keys() | injected.keys() if base.get(key) != injected.get(key)}
    assert differences == set(ALLOWED), "params.injected.yaml must mirror params.yaml (copy the change over)"
    for key, value in ALLOWED.items():
        assert injected[key] == value
    assert base[("inject_issues", "enabled")] is False and base[("validation", "mode")] == "strict"
