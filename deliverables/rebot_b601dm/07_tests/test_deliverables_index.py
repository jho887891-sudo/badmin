"""The deliverables index must not reference files that do not exist."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

INDEX = Path("management/rebot_b601dm/DELIVERABLES.md")


@pytest.fixture(scope="module")
def text():
    if not INDEX.is_file():
        pytest.skip("deliverables index not generated")
    return INDEX.read_text(encoding="utf-8")


def test_every_listed_file_exists(text):
    """A stale index that names a deleted artifact is worse than no index."""
    # only the FIRST backticked cell on each row is a path; the sha256 column is backticked too
    listed = re.findall(r"^\| \x60([^\x60]+)\x60", text, re.M)
    assert listed, "the index should list files"
    missing = [rel for rel in listed if not Path(rel).is_file()]
    assert not missing, "the index names files that are not there: " + ", ".join(missing)


def test_it_states_that_the_delivery_is_not_accepted(text):
    """Gate 3 has no measurement, and the index must not imply otherwise."""
    assert "UNEVALUATED" in text
    assert "not accepted" in text.lower()


def test_it_names_what_is_missing(text):
    for expected in ("racket-head speed", "V4", "angle limits", "racket mass"):
        assert expected in text, expected


def test_it_gives_the_way_to_complete_the_missing_gate(text):
    assert "measure_racket_speed.py" in text
    assert "wait_for_quiet_window.sh" in text


def test_it_can_be_regenerated(text):
    """The generator must run; a broken generator makes the index unmaintainable."""
    r = subprocess.run([sys.executable, "tools/generate_b601dm_deliverables.py"],
                       capture_output=True, timeout=180, cwd=".")
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[:400]
    assert "files present" in r.stdout.decode("utf-8", "replace")


def test_the_regenerated_index_still_lists_only_existing_files():
    after = INDEX.read_text(encoding="utf-8")
    listed = re.findall(r"^\| \x60([^\x60]+)\x60", after, re.M)
    assert not [rel for rel in listed if not Path(rel).is_file()]
