from eval_harness import run_tier1, run_tier2


def test_tier1_gold_benchmark_has_no_regressions():
    report = run_tier1()
    assert report.ok, [r for r in report.results if not r.passed]


def test_tier2_meets_threshold_and_judge_is_calibrated():
    report = run_tier2()
    assert report.ok, report.model_dump()
    assert report.false_negatives == 0
