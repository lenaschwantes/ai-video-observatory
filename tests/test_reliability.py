"""Failure modes found in the reliability audit, each pinned by a test."""

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from observatory.bench.metrics import auc, score
from observatory.detect import pipeline, web
from observatory.learn import feedback, weights


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(web, "BENCHMARKS_FILE", tmp_path / "none.json")
    web._usage.clear()
    return TestClient(web.app, raise_server_exceptions=False)


# --- corrupt files must not take pages down ------------------------------------


def test_a_truncated_report_is_skipped_not_fatal(client, tmp_path):
    (tmp_path / "youtube-x").mkdir()
    (tmp_path / "youtube-x" / "report.json").write_text('{"slug": "youtube-x", "trunc')
    assert client.get("/").status_code == 200
    assert client.get("/report/youtube-x").status_code == 404
    assert pipeline.load_report(tmp_path, "youtube-x") is None


def test_report_writes_are_atomic(tmp_path):
    target = tmp_path / "report.json"
    pipeline._write_atomic(target, '{"ok": true}')
    assert json.loads(target.read_text()) == {"ok": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_broken_weights_fall_back_to_v0(tmp_path):
    (tmp_path / "current.json").write_text('{"version": "v7"}')  # v7.json missing
    assert weights.load_current(tmp_path).version == weights.DEFAULT.version
    (tmp_path / "current.json").write_text("not json")
    assert weights.load_current(tmp_path).version == weights.DEFAULT.version


# --- metrics must refuse mismatched inputs ---------------------------------------


def test_mismatched_labels_and_scores_raise_instead_of_truncating():
    with pytest.raises(ValueError):
        score([1, 1, 0, 0], [0.9, 0.8])
    with pytest.raises(ValueError):
        auc([1, 0, 1], [0.9, 0.1])


# --- server under load -------------------------------------------------------------


def test_finished_jobs_are_pruned(monkeypatch):
    web._jobs.clear()
    web._jobs["old"] = {"stage": "done", "ended": 1000.0}
    web._jobs["fresh"] = {"stage": "done", "ended": 1000.0 + web.JOB_TTL}
    web._jobs["running"] = {"stage": "visual", "ended": None}
    web._prune_jobs(now=1000.0 + web.JOB_TTL + 1)
    assert set(web._jobs) == {"fresh", "running"}


def test_busy_server_refuses_politely_and_frees_the_slot(client, monkeypatch):
    monkeypatch.setattr(web, "_job_slots", threading.BoundedSemaphore(1))
    web._job_slots.acquire()  # someone else is analyzing
    resp = client.post("/analyze", data={"url": "https://www.youtube.com/watch?v=x"}, headers={"HX-Request": "true"})
    assert "ocupado" in resp.text or "busy" in resp.text
    web._job_slots.release()
    assert web._job_slots.acquire(blocking=False)  # the refusal did not leak a slot
    web._job_slots.release()


def test_slot_is_released_after_a_job_finishes(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web, "_job_slots", threading.BoundedSemaphore(1))
    monkeypatch.setattr(web, "_run", lambda url, upload, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    client.post("/analyze", data={"url": "https://www.youtube.com/watch?v=x"}, headers={"HX-Request": "true"})
    for _ in range(100):
        if web._job_slots.acquire(blocking=False):
            web._job_slots.release()
            break
        time.sleep(0.01)
    else:
        pytest.fail("job slot never released after a failed analysis")


def test_oversized_requests_are_refused_from_the_header(client):
    resp = client.post("/analyze", content=b"x", headers={"content-length": str(web.MAX_REQUEST_BYTES + 1),
                                                          "content-type": "multipart/form-data; boundary=x"})
    assert resp.status_code == 413


def test_same_video_runs_one_at_a_time():
    lock = pipeline._lock_for("youtube-x")
    assert lock is pipeline._lock_for("youtube-x") and lock is not pipeline._lock_for("youtube-y")


# --- votes and curated labels (learning from use) ---------------------------------


def _vote(slug="youtube-x", voter="v1", claimed="ai", p=0.5):
    return feedback.Vote(slug=slug, claimed=claimed, voter=voter, model_verdict="inconclusive", model_probability=p)


def test_one_vote_per_voter_per_report(tmp_path):
    assert feedback.add_vote(_vote(), tmp_path)
    assert not feedback.add_vote(_vote(claimed="real"), tmp_path)
    assert feedback.add_vote(_vote(voter="v2"), tmp_path)
    assert len(feedback.votes(tmp_path)) == 2


def test_voters_are_hashed_never_stored_as_ip():
    vid = feedback.voter_id("203.0.113.7")
    assert "203.0.113.7" not in vid and len(vid) == 16 and vid == feedback.voter_id("203.0.113.7")


def test_evidence_links_must_be_http():
    with pytest.raises(ValueError):
        feedback.Vote(slug="s", claimed="ai", voter="v", model_verdict="x", model_probability=0.5,
                      evidence_url="javascript:alert(1)")


def test_queue_puts_uncertain_and_disputed_first_and_skips_curated(tmp_path):
    reports = {"sure": {"probability": 0.95}, "unsure": {"probability": 0.5}, "disputed": {"probability": 0.9},
               "done": {"probability": 0.5}}
    feedback.add_vote(_vote(slug="disputed", claimed="real"), tmp_path)
    feedback.add_label(feedback.Label(slug="done", label="real", source="uploaded in 2015"), tmp_path)
    order = [r["slug"] for r in feedback.queue(reports, tmp_path)]
    assert order[0] in ("unsure", "disputed") and order[-1] == "sure" and "done" not in order


def test_latest_curated_label_wins(tmp_path):
    feedback.add_label(feedback.Label(slug="s", label="ai", source="first look"), tmp_path)
    feedback.add_label(feedback.Label(slug="s", label="real", source="creator confirmed it is real"), tmp_path)
    assert feedback.labels(tmp_path)["s"].label == "real"


# --- versioned weights ----------------------------------------------------------------


def test_weights_adopt_record_and_reload(tmp_path):
    w = weights.Weights(version=weights.next_version(tmp_path), bias=-0.2, per_signal={"visual": 0.5})
    weights.adopt(w, tmp_path)
    weights.record({"version": w.version, "adopted": True}, tmp_path)
    loaded = weights.load_current(tmp_path)
    assert loaded.version == "v1" and loaded.weight("visual") == 0.5 and loaded.weight("text") == 1.0
    assert weights.next_version(tmp_path) == "v2"
    assert weights.history(tmp_path) == [{"version": "v1", "adopted": True}]


def test_combine_applies_the_weights():
    from observatory.detect.models import Signal

    sig = [Signal(name="visual", label="v", available=True, llr=2.0)]
    full = pipeline.combine(sig)[0]
    halved = pipeline.combine(sig, weights.Weights(version="t", per_signal={"visual": 0.5}))[0]
    assert halved < full
