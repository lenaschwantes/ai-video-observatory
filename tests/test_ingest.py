import json
from datetime import date

import pytest

from observatory.ingest import IngestError, ingest
from observatory.sources import FixtureSource, SourceUnavailable

DAY = date(2026, 10, 6)  # the fixture day with one malformed row


class BrokenSource:
    name = "broken"

    def fetch(self, captured_on, limit):
        raise SourceUnavailable("blocked")

    def normalize(self, payload, captured_on):  # pragma: no cover
        raise AssertionError("never called")


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_lands_raw_and_normalized_side_by_side(tmp_path):
    result = ingest([FixtureSource()], DAY, tmp_path)
    rows = read_lines(result.path)
    assert result.path == tmp_path / "dt=2026-10-06" / "fixture.jsonl"
    assert len(rows) == result.accepted
    assert rows[0]["payload"]["id"] == rows[0]["normalized"]["video_id"]


def test_contract_violations_are_quarantined(tmp_path):
    result = ingest([FixtureSource()], DAY, tmp_path)
    assert result.rejected == 1
    rejected = read_lines(tmp_path / "dt=2026-10-06" / "_rejected" / "fixture.jsonl")
    assert rejected[0]["payload"]["stats"]["playCount"] == -1
    assert "greater than or equal to 0" in rejected[0]["error"]


def test_rerun_is_idempotent(tmp_path):
    first = ingest([FixtureSource()], DAY, tmp_path)
    second = ingest([FixtureSource()], DAY, tmp_path)
    assert len(read_lines(second.path)) == first.accepted


def test_falls_back_to_next_source(tmp_path):
    result = ingest([BrokenSource(), FixtureSource()], DAY, tmp_path)
    assert result.source == "fixture"
    assert result.failed_sources == [("broken", "blocked")]


def test_raises_when_every_source_fails(tmp_path):
    with pytest.raises(IngestError, match="broken: blocked"):
        ingest([BrokenSource()], DAY, tmp_path)
