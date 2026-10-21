#!/usr/bin/env python
"""A03: queue/repo consistency gate.

Fails when the recorded queue contradicts the repository:

1. structural problems (schema, illegal status, cycle, unknown dependency);
2. a DONE task without artifact + head_sha, or with an artifact/sha the repo
   cannot resolve (file missing in the worktree and in that commit; sha not a
   reachable commit);
3. two tasks claiming the same artifact "concurrently" (any RUNNING/DONE pair);
4. a run-level claim that is not backed by a complete run directory
   (manifest.json + result.json + raw_poses.jsonl);
5. a RUNNING/PARTIAL task whose latest_run_id does not exist in the repo at all;
6. the committed task registry disagreeing with the dependency policy.

Exit codes: 0 = consistent, 1 = problems found, 3 = fail-closed internal error.

Pure CPU, python standard library only; never starts Isaac/Kit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.queue import queue_state as qs  # noqa: E402

SCHEMA_VERSION = "1.0"
EVIDENCE_ROOT = "outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates"
RUN_REQUIRED_FILES = ("manifest.json", "result.json", "raw_poses.jsonl")


class RepoView:
    """Small cached view of the repository used by the checks."""

    def __init__(self, repo: Path):
        self.repo = repo
        self._trees = {}
        self._commits = {}

    def commit_exists(self, sha: str) -> bool:
        if not sha:
            return False
        if sha not in self._commits:
            self._commits[sha] = qs.sha_exists(self.repo, sha)
        return self._commits[sha]

    def tree(self, sha: str):
        if sha not in self._trees:
            try:
                out = qs._git(self.repo, "ls-tree", "-r", "--name-only", sha)
                self._trees[sha] = set(out.splitlines())
            except RuntimeError:
                self._trees[sha] = set()
        return self._trees[sha]

    def resolve(self, artifact: str, sha):
        """Where the artifact really exists: 'worktree', 'git:<sha>', or None."""
        if not artifact:
            return None
        local = self.repo / artifact
        if local.exists():
            return "worktree"
        if sha and self.commit_exists(sha):
            tree = self.tree(sha)
            prefix = artifact.rstrip("/") + "/"
            if artifact in tree or any(name.startswith(prefix) for name in tree):
                return "git:%s" % sha[:12]
        return None

    def run_dirs_at(self, sha: str):
        if not sha or not self.commit_exists(sha):
            return set()
        prefix = EVIDENCE_ROOT + "/"
        runs = set()
        for name in self.tree(sha):
            if name.startswith(prefix):
                rest = name[len(prefix):]
                if "/" in rest:
                    runs.add(rest.split("/")[0])
        return runs

    def local_run_dirs(self):
        root = self.repo / EVIDENCE_ROOT
        if not root.is_dir():
            return set()
        return {p.name for p in root.iterdir() if p.is_dir()}


# --------------------------------------------------------------------- checks


def check_structure(state: dict, repo: Path):
    problems = list(qs.validate_state(state))
    try:
        qs.compute_queue(state)
    except qs.QueueCycleError as exc:
        problems.append("FAIL CLOSED: %s" % exc)
    return problems


def check_registry(state: dict, repo: Path, registry_path: Path):
    problems = []
    if not registry_path.is_file():
        return ["registry snapshot missing: %s" % registry_path.name]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    state_ids = [t["task_id"] for t in state["tasks"]]
    registry_ids = [t["task_id"] for t in registry.get("tasks", [])]
    if state_ids != registry_ids:
        missing = sorted(set(registry_ids) - set(state_ids))
        extra = sorted(set(state_ids) - set(registry_ids))
        problems.append(
            "state and registry disagree (missing=%s, extra=%s)" % (missing, extra)
        )
    for task in registry.get("tasks", []):
        declared = list(task.get("dependencies") or [])
        policy = qs.dependencies_for(task["task_id"])
        if declared != policy:
            problems.append(
                "%s: registry dependencies %s disagree with the policy %s"
                % (task["task_id"], declared, policy)
            )
    state_deps = {t["task_id"]: list(t.get("dependencies") or []) for t in state["tasks"]}
    for task_id, deps in state_deps.items():
        if deps != qs.dependencies_for(task_id):
            problems.append(
                "%s: state dependencies %s disagree with the policy %s"
                % (task_id, deps, qs.dependencies_for(task_id))
            )
    return problems


def check_done_evidence(state: dict, view: RepoView):
    problems = []
    for task in state["tasks"]:
        task_id = task["task_id"]
        status = task.get("status")
        artifact = task.get("artifact")
        sha = task.get("head_sha")
        if status == qs.DONE:
            if not artifact:
                problems.append("%s: DONE without artifact" % task_id)
            if not sha:
                problems.append("%s: DONE without head_sha" % task_id)
            if artifact and not view.resolve(artifact, sha):
                where = "the worktree or commit %s" % (sha or "<none>")
                problems.append(
                    "%s: DONE artifact %r does not exist in %s" % (task_id, artifact, where)
                )
            if sha and not view.commit_exists(sha):
                problems.append("%s: DONE head_sha %r is not a reachable commit" % (task_id, sha))
            if not task.get("owner"):
                problems.append("%s: DONE without owner" % task_id)
        elif status in ("PARTIAL", "RUNNING", "FAILED"):
            if not artifact:
                problems.append("%s: %s without artifact" % (task_id, status))
            elif not view.resolve(artifact, sha):
                problems.append(
                    "%s: %s artifact %r is not resolvable (worktree or commit)"
                    % (task_id, status, artifact)
                )
            if sha and not view.commit_exists(sha):
                problems.append("%s: head_sha %r is not a reachable commit" % (task_id, sha))
    return problems


def check_duplicate_claims(state: dict):
    problems = []
    claims = {}
    for task in state["tasks"]:
        artifact = task.get("artifact")
        if not artifact:
            continue
        claims.setdefault(artifact.rstrip("/"), []).append(task)
    for artifact, tasks in sorted(claims.items()):
        if len(tasks) < 2:
            continue
        statuses = {t.get("status") for t in tasks}
        ids = sorted(t["task_id"] for t in tasks)
        if statuses & {"RUNNING", "DONE"}:
            problems.append(
                "duplicate artifact claim on %s by %s (statuses %s)"
                % (artifact, ", ".join(ids), sorted(statuses))
            )
    return problems


def check_run_integrity(state: dict, view: RepoView):
    problems = []
    for task in state["tasks"]:
        artifact = task.get("artifact") or ""
        if not artifact.startswith(EVIDENCE_ROOT + "/"):
            continue
        rest = artifact[len(EVIDENCE_ROOT) + 1:]
        run_id = rest.split("/")[0]
        run_prefix = "%s/%s/" % (EVIDENCE_ROOT, run_id)
        sha = task.get("head_sha")
        available = set()
        local = view.repo / (EVIDENCE_ROOT + "/" + run_id)
        if local.is_dir():
            available = {p.name for p in local.iterdir() if p.is_file()}
        elif sha and view.commit_exists(sha):
            available = {
                name[len(run_prefix):]
                for name in view.tree(sha)
                if name.startswith(run_prefix) and "/" not in name[len(run_prefix):]
            }
        if not available:
            problems.append("%s: run directory %s/%s is missing" % (task["task_id"], EVIDENCE_ROOT, run_id))
            continue
        for required in RUN_REQUIRED_FILES:
            if required not in available:
                problems.append(
                    "%s: run %s is incomplete, missing %s" % (task["task_id"], run_id, required)
                )
    return problems


def check_run_ids(state: dict, view: RepoView):
    problems = []
    local_runs = view.local_run_dirs()
    cache = {}
    for task in state["tasks"]:
        run_id = task.get("latest_run_id")
        status = task.get("status")
        if not run_id or status not in ("RUNNING", "PARTIAL", "FAILED", "DONE"):
            continue
        if run_id in local_runs:
            continue
        sha = task.get("head_sha")
        if sha not in cache:
            cache[sha] = view.run_dirs_at(sha)
        if run_id in cache[sha]:
            continue
        problems.append(
            "%s: latest_run_id %r exists neither in the worktree nor in commit %s"
            % (task["task_id"], run_id, sha)
        )
    return problems


def check_consistency(state: dict, repo: Path, registry_path=None) -> "list[str]":
    view = RepoView(repo)
    registry_path = registry_path or (repo / qs.REGISTRY_PATH)
    problems = []
    problems += check_structure(state, repo)
    if not any(p.startswith("FAIL CLOSED") for p in problems):
        problems += check_registry(state, repo, Path(registry_path))
    else:
        return problems
    problems += check_done_evidence(state, view)
    problems += check_duplicate_claims(state)
    problems += check_run_integrity(state, view)
    problems += check_run_ids(state, view)
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="A03 queue/repo consistency gate (pure CPU).")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--state", default=qs.STATE_PATH)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    repo = qs.repo_root(args.repo)
    state_path = repo / args.state
    if not state_path.is_file():
        print("error: queue state not found: %s" % args.state, file=sys.stderr)
        return 3
    try:
        state = qs.load_state(state_path)
        problems = check_consistency(state, repo, repo / args.registry if args.registry else None)
    except qs.QueueCycleError as exc:
        print("FAIL CLOSED: %s" % exc, file=sys.stderr)
        return 3
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps({"consistent": not problems, "problems": problems},
                         ensure_ascii=False, indent=2))
    else:
        for problem in problems:
            print("INCONSISTENT %s" % problem)
        if problems:
            print("%d problem(s)" % len(problems))
        else:
            print("consistent: %d slots checked" % len(state["tasks"]))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
