from eventstudy.features import FEATURE_RELATIVE_RANGES, assert_no_feature_leakage


def test_no_feature_uses_data_after_day0():
    assert_no_feature_leakage()
    assert all(end <= 0 for _, end in FEATURE_RELATIVE_RANGES.values())
