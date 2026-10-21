"""Tests for the A01 engineering index and the A10 pure-CPU contract."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from tools.queue import a01_engineering_index as a01

REPO = Path(__file__).resolve().parents[2]
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def index():
    return a01.build_index(REPO)


def test_index_has_schema_and_generator(index):
    assert index["schema_version"] == a01.SCHEMA_VERSION
    assert index["generator"] == "tools/queue/a01_engineering_index.py"
    assert index["spec"]["path"].endswith(".md")
    assert index["environment"]["pure_cpu"] is True
    assert index["environment"]["launches_isaac_or_kit"] is False


def test_index_branch_head_shas_match_git(index):
    assert index["branch_count"] == len(index["branches"])
    assert index["branch_count"] > 0
    for branch in index["branches"]:
        assert SHA_RE.match(branch["head_sha"]), branch
        assert branch["name"]
        assert isinstance(branch["pull_requests"], list)
    sample = index["branches"][:3]
    for branch in sample:
        code, out = a01.git(REPO, "rev-parse", branch["name"])
        assert code == 0
        assert out.strip() == branch["head_sha"]


def test_index_records_worktrees_and_pulls(index):
    assert index["worktree_count"] == len(index["worktrees"])
    paths = [item.get("path") for item in index["worktrees"]]
    assert any(Path(path).resolve() == REPO for path in paths), paths
    assert index["pull_count"] == len(index["pulls"])


def test_index_documents_have_sha256(index):
    assert index["documents"], "expected management/docs documents"
    for document in index["documents"][:5]:
        assert SHA256_RE.match(document["sha256"])
        assert document["bytes"] > 0


def test_index_marks_default_branch_honestly(index):
    if index["repo"]["default_branch"] is None:
        assert any("origin/HEAD is not set" in note for note in index["notes"])


def test_index_evidence_refs_are_run_shaped(index):
    for run in index["evidence_from_refs"][:5]:
        assert run["run_id"]
        assert run["source"].startswith("git:")
        assert isinstance(run["files"], list)


def test_strip_volatile_is_deterministic(index):
    stripped = a01.strip_volatile(index)
    assert "generated_at_utc" not in stripped
    assert "branches" not in stripped  # ref-shaped facts are compared separately
    assert stripped == a01.strip_volatile(index)


def test_strip_moving_keeps_the_refs(index):
    moving = a01.strip_moving(index)
    assert "branches" in moving and "worktrees" in moving
    assert "repo_head_sha" in moving
    assert "generated_at_utc" not in moving


def test_stable_view_ignores_the_worktree_head(index):
    moved = json.loads(json.dumps(index))
    moved["repo"]["head_sha"] = "0" * 40
    moved["repo"]["head_branch"] = "someone/else"
    assert a01.strip_volatile(moved) == a01.strip_volatile(index)


def test_stable_view_changes_when_a_document_changes(index):
    tampered = json.loads(json.dumps(index))
    tampered["documents"][0]["sha256"] = "0" * 64
    assert a01.strip_volatile(tampered) != a01.strip_volatile(index)


def test_index_write_and_check(tmp_path, capsys):
    out = tmp_path / "index.json"
    assert a01.main(["--repo", str(REPO), "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == a01.SCHEMA_VERSION
    assert a01.main(["--repo", str(REPO), "--out", str(out), "--check"]) == 0
    data["documents"] = []
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    assert a01.main(["--repo", str(REPO), "--out", str(out), "--check"]) == 1


def test_index_check_is_stable_while_refs_move(tmp_path, capsys):
    """A colleague pushing a branch must not turn --check red (non-strict)."""
    out = tmp_path / "index.json"
    assert a01.main(["--repo", str(REPO), "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    data["branch_count"] = -1
    data["branches"] = []
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    assert a01.main(["--repo", str(REPO), "--out", str(out), "--check"]) == 0
    assert a01.main(["--repo", str(REPO), "--out", str(out), "--check", "--strict"]) == 1


# --------------------------------------------------------- A10 pure CPU proof

TOOL_DIR = REPO / "tools" / "queue"
FORBIDDEN = ("isaacsim", "isaaclab", "isaac", "omni", "pxr", "torch", "carb", "numpy")
STDLIB_ALLOWED = {
    "__future__", "argparse", "ast", "hashlib", "json", "os", "re", "subprocess", "sys",
    "tempfile", "pathlib", "datetime", "typing", "collections", "unittest",
    # A05/A06/A09 additions: standard library only, none of them touches Isaac/Kit
    "platform", "importlib", "shlex",
}


def tool_sources():
    return sorted(TOOL_DIR.glob("*.py"))


def test_tools_are_stdlib_only():
    assert tool_sources(), "no tool sources found"
    for path in tool_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module.split(".")[0]]
            for module in modules:
                assert module in STDLIB_ALLOWED or module in ("tools",), (path.name, module)
                assert module.lower() not in FORBIDDEN, (path.name, module)


def test_tools_never_launch_a_simulator():
    """Only git/gh may be spawned; no argument may name Isaac/Kit."""
    for path in tool_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "attr", getattr(func, "id", ""))
            if name not in ("run", "Popen", "check_output", "subprocess"):
                continue
            for argument in node.args:
                if isinstance(argument, (ast.List, ast.Tuple)):
                    for element in argument.elts:
                        if isinstance(element, ast.Constant) and isinstance(element.value, str):
                            lowered = element.value.lower()
                            assert "isaac" not in lowered, (path.name, element.value)
                            assert "kit" != lowered and "isaac-sim" not in lowered, (path.name, element.value)


def test_readme_documents_cpu_only_commands():
    readme = REPO / "docs" / "queue" / "README_zh.md"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")
    assert "python -m pytest tests/queue" in text
    assert "tools/queue/queue_state.py" in text
    assert "tools/queue/check_consistency.py" in text
    assert "不使用 Isaac" in text or "不启动 Isaac" in text
