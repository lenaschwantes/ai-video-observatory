from datetime import date, timedelta

from observatory.trends import detect_emerging, robust_zscore

START = date(2026, 10, 1)


def series(values):
    return [(START + timedelta(days=i), float(v)) for i, v in enumerate(values)]


def test_flags_breakout_on_latest_day():
    flagged = detect_emerging(
        {
            "breakout": series([100, 110, 95, 105, 98, 102, 101, 900]),
            "steady": series([100, 110, 95, 105, 98, 102, 101, 108]),
        }
    )
    assert [e.tag for e in flagged] == ["breakout"]
    assert flagged[0].day == START + timedelta(days=7)


def test_one_old_spike_does_not_inflate_the_baseline():
    # mean/std would absorb the old spike and miss the new one; median/MAD does not
    assert robust_zscore([100, 105, 5000, 98, 102, 101, 99], 400) > 3.5


def test_skips_tags_without_enough_history():
    assert detect_emerging({"new": series([10, 5000])}) == []


def test_flat_history_only_flags_increases():
    assert robust_zscore([50] * 7, 50) == 0.0
    assert robust_zscore([50] * 7, 80) == float("inf")
