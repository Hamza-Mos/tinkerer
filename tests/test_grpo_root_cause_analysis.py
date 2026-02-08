"""Tests for GRPO root cause analysis heuristics.

These guard against regressions where GRPO "no-signal" skips (uniform rewards)
are misclassified as fatal errors, especially when dominated by ALL-1.0
(which usually indicates the model is succeeding consistently).
"""


def test_uniform_all_ones_reports_success():
    from tinker_mcp.training.grpo import FailureStats, _get_root_cause_analysis

    failures = FailureStats()
    failures.uniform_reward_samples = 80
    failures.uniform_all_ones = 80
    failures.uniform_all_zeros = 0
    failures.samples_included = 20
    failures.prompts_processed = 25
    failures.prompts_with_no_signal = 20

    analysis, fix = _get_root_cause_analysis(failures)
    assert analysis.startswith("[SUCCESS]")
    assert "ALL 1s" in analysis
    assert "May be reaching ceiling" in fix


def test_uniform_all_zeros_reports_critical():
    from tinker_mcp.training.grpo import FailureStats, _get_root_cause_analysis

    failures = FailureStats()
    failures.uniform_reward_samples = 80
    failures.uniform_all_ones = 0
    failures.uniform_all_zeros = 80
    failures.samples_included = 20
    failures.prompts_processed = 25
    failures.prompts_with_no_signal = 20

    analysis, _fix = _get_root_cause_analysis(failures)
    assert analysis.startswith("[CRITICAL]")
    assert "ALL 0s" in analysis
