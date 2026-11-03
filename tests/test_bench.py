import json

import pytest

from observatory.bench import runner
from observatory.bench.metrics import auc, ece, score
from observatory.detect.models import Frame
from observatory.detect.vision import Usage, VisualAssessment


def test_auc_perfect_random_and_ties():
    assert auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert auc([1, 0], [0.5, 0.5]) == 0.5
    assert auc([1, 1], [0.9, 0.8]) is None


def test_ece_is_zero_when_confidence_matches_frequency():
    assert ece([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)
    assert ece([0, 0], [0.9, 0.9]) == pytest.approx(0.9)


def test_score_counts_false_positives_on_real_videos_only():
    s = score([1, 1, 0, 0], [0.9, 0.5, 0.8, 0.1])
    assert s.recall_ai == 0.5
    assert s.false_positive_rate == 0.5
    assert s.inconclusive_rate == 0.25
    assert s.accuracy_when_decided == pytest.approx(2 / 3)


def _cache(tmp_path, vid):
    d = tmp_path / "cache" / vid
    d.mkdir(parents=True)
    (d / "frames.json").write_text(json.dumps({"frames": [Frame(index=0, timestamp=1.0, file="f.jpg").model_dump()]}))


def test_run_resumes_and_report_writes_docs(tmp_path):
    rows = [
        {"id": "a1", "label": "ai", "height": "1080"},
        {"id": "r1", "label": "real", "height": "360"},
        {"id": "r2", "label": "real", "height": "1080"},
    ]
    for r in rows:
        _cache(tmp_path, r["id"])
    calls = []

    def fake(provider, model, frames, frame_dir):
        calls.append(frame_dir.name)
        if frame_dir.name == "r2":
            raise RuntimeError("rate limited")
        p = 0.9 if frame_dir.name == "a1" else 0.1
        return VisualAssessment(ai_likelihood=p, findings=[], summary=""), Usage(100, 20, 1, 0.5)

    kw = dict(cache=tmp_path / "cache", results=tmp_path / "results", assess_with=fake)
    runner.run("groq", "m/x", rows, **kw)
    runner.run("groq", "m/x", rows, **kw)  # nothing new: errors are not retried by default
    assert calls == ["a1", "r1", "r2"]
    runner.run("groq", "m/x", rows, retry_errors=True, **kw)
    assert calls[-1] == "r2"

    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"groq/m/x": [1.0, 2.0]}))
    [summary] = runner.report(rows, results=tmp_path / "results", docs=tmp_path / "docs", prices_file=prices)
    assert summary["failures"] == 1 and summary["scores"]["auc"] == 1.0
    assert summary["cost_per_100_videos_usd"] == pytest.approx(0.014)
    assert "groq / `m/x`" in (tmp_path / "docs" / "benchmarks.md").read_text(encoding="utf-8")


def test_find_ai_score_handles_class_lists_and_keys():
    from observatory.bench.detectors import find_ai_score

    assert find_ai_score({"output": [{"classes": [{"class": "not_ai_generated", "score": 0.1}, {"class": "ai_generated", "score": 0.9}]}]}) == 0.9
    assert find_ai_score({"data": {"ai_generated": 0.3}}) == 0.3
    assert find_ai_score({"data": {"ai_generated": True}}) is None
    assert find_ai_score({"nothing": 1}) is None


def test_detector_scores_frames_and_merges(tmp_path, monkeypatch):
    from observatory.bench import detectors

    monkeypatch.setitem(detectors.SCORERS, "sightengine", lambda path: 0.9 if path.name == "a.jpg" else 0.6)
    frames = [Frame(index=0, timestamp=0, file="a.jpg"), Frame(index=1, timestamp=1, file="b.jpg")]
    result, usage = detectors.assess_with("sightengine", "genai", frames, tmp_path)
    assert 0.6 < result.ai_likelihood < 0.9 and usage.calls == 2
    assert [f.direction for f in result.findings] == ["ai", "neutral"]
