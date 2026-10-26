import copy

import pytest

from nsh_pipeline.bias.report import write_report
from nsh_pipeline.bias.slicing import analyse


@pytest.fixture(scope="module")
def injected_result(injected, injected_params):
    return analyse(injected[0], injected_params["bias"])


def test_report_explains_flag_mitigation_and_team_assumptions(injected_result, injected_params, tmp_path):
    report = write_report(injected_result, injected_params, tmp_path).read_text(encoding="utf-8")
    assert "**Flagged: state = CA**" in report
    assert "`denial_skew` flipped 30% of approved swipes in state CA to Denied. Detected." in report
    assert "Reweighing on `state`" in report and "Trade-off" in report
    assert "is_vip = True (n = 10)" in report  # too small to judge
    for decision in ("decision #3", "decision #4", "decision #10"):  # assumptions M5 must mention
        assert decision in report
    assert report.index("group_size_bucket = 25-49") < report.index("group_size_bucket = 100-249")  # size order


def test_report_says_when_the_planted_bias_is_missed(injected_result, injected_params, tmp_path):
    params = copy.deepcopy(injected_params)
    params["inject_issues"]["denial_skew"]["state"] = "NV"
    report = write_report(injected_result, params, tmp_path).read_text(encoding="utf-8")
    assert "state NV to Denied. **Not detected.**" in report
