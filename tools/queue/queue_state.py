#!/usr/bin/env python
"""EXECUTION_QUEUE_STATE.json: schema, generator, dependency topology and the
automatic READY selector for the 96-slot P0-P8 execution queue.

Design rules (SPEC: management/morph_one_rs/EXECUTION_QUEUE_CONTINUOUS_P0_P8.md):

* The queue file is data, not a promise: a slot is DONE only with an existing
  artifact plus a reachable commit SHA.
* READY is derived: every dependency must be DONE.
* Cycles fail closed; BLOCKED/FAILED propagate with an explicit chain.
* Writes are atomic (temp file + os.replace); DONE evidence is immutable.

Pure CPU, python standard library only. Never imports Isaac/Kit/omni/pxr/torch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA_VERSION = "1.0"
GENERATOR = "tools/queue/queue_state.py"

STATUSES = ("READY", "RUNNING", "DONE", "PARTIAL", "FAILED", "BLOCKED", "UNKNOWN")
DONE = "DONE"
EXPECTED_TASK_COUNT = 96

STATE_PATH = "management/morph_one_rs/EXECUTION_QUEUE_STATE.json"
REGISTRY_PATH = "management/morph_one_rs/QUEUE_TASK_REGISTRY.json"
SPEC_PATH = "management/morph_one_rs/EXECUTION_QUEUE_CONTINUOUS_P0_P8.md"
DEFAULT_SPEC_REF = "origin/chatgpt/p2-wc-continuous-queue-20261021"

REQUIRED_TASK_FIELDS = (
    "task_id",
    "stage",
    "title",
    "dependencies",
    "status",
    "owner",
    "head_sha",
    "artifact",
    "latest_run_id",
    "next_action",
    "blocker",
    "last_error",
)


# ---------------------------------------------------------------- repo helpers


def repo_root(start=None) -> Path:
    """Repository root: explicit override, else two levels above this file."""
    if start is not None:
        return Path(start).resolve()
    return Path(__file__).resolve().parents[2]


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "git %s failed (%d): %s" % (" ".join(args), proc.returncode, proc.stderr.strip())
        )
    return proc.stdout.strip()


def sha_exists(repo: Path, sha: str) -> bool:
    """True when sha is a commit object that some ref (or HEAD) reaches."""
    if not sha or not re.fullmatch(r"[0-9a-fA-F]{7,40}", sha.strip()):
        return False
    try:
        _git(repo, "cat-file", "-e", "%s^{commit}" % sha)
    except RuntimeError:
        return False
    try:
        refs = _git(repo, "for-each-ref", "--contains", sha, "--format=%(refname)")
    except RuntimeError:
        refs = ""
    if refs.strip():
        return True
    try:
        head = _git(repo, "rev-parse", "HEAD")
    except RuntimeError:
        return False
    return head.startswith(sha) or sha.startswith(head)


# --------------------------------------------------------------- spec parsing

TASK_LINE = re.compile(
    r"^\s*-\s*\[[ xX]?\]\s*\*\*(?P<task_id>[A-Z]{1,3}[0-9]{1,2}(?:-[0-9]{2})?)\*\*\s*[：:]\s*(?P<title>.+?)\s*$"
)


class SpecError(RuntimeError):
    pass


def artifact_present(repo: Path, artifact: str, sha=None) -> bool:
    """Artifact exists in the worktree, or in the commit that produced it."""
    if not artifact:
        return False
    if (repo / artifact).exists():
        return True
    if not sha or not sha_exists(repo, sha):
        return False
    try:
        prefix = artifact.rstrip("/") + "/"
        listing = _git(repo, "ls-tree", "-r", "--name-only", sha).splitlines()
    except RuntimeError:
        return False
    return artifact in listing or any(name.startswith(prefix) for name in listing)


def stage_of(task_id: str) -> str:
    head = task_id.split("-")[0]
    if "-" in task_id:
        return head
    return re.sub(r"[0-9]+$", "", head)


def read_spec(repo: Path, spec_ref=None, spec_path: str = SPEC_PATH):
    """Return (markdown text, source metadata).

    Prefers the working-tree file; falls back to "git show ref:path" so a fresh
    checkout of any branch can still regenerate the queue.
    """
    local = repo / spec_path
    if local.is_file():
        text = local.read_text(encoding="utf-8")
        try:
            sha = _git(repo, "log", "-1", "--format=%H", "--", spec_path)
        except RuntimeError:
            sha = None
        return text, {"kind": "worktree", "path": spec_path, "ref": None, "sha": sha}

    ref = spec_ref or DEFAULT_SPEC_REF
    try:
        text = _git(repo, "show", "%s:%s" % (ref, spec_path))
    except RuntimeError as exc:
        raise SpecError(
            "cannot read the queue spec from %s or %s (%s)" % (spec_path, ref, exc)
        ) from exc
    try:
        sha = _git(repo, "rev-parse", ref)
    except RuntimeError:
        sha = None
    return text, {"kind": "git", "path": spec_path, "ref": ref, "sha": sha}


def parse_spec(text: str):
    """Parse '- [ ] **P2-01**：title' lines into task slots."""
    tasks = []
    seen = set()
    for line in text.splitlines():
        match = TASK_LINE.match(line)
        if not match:
            continue
        task_id = match.group("task_id")
        if task_id in seen:
            raise SpecError("duplicate task id in spec: %s" % task_id)
        seen.add(task_id)
        tasks.append(
            {
                "task_id": task_id,
                "stage": stage_of(task_id),
                "title": match.group("title").strip(),
            }
        )
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise SpecError("expected %d task slots, parsed %d" % (EXPECTED_TASK_COUNT, len(tasks)))
    return tasks


# ------------------------------------------------- dependency policy (SPEC 3)


def dependencies_for(task_id: str):
    """Hard dependencies, transcribed from SPEC section 3 dispatch rules."""
    if task_id.startswith("P3-"):
        return ["P2-10"]
    if task_id.startswith("P4-"):
        return ["P1-09", "P2-10", "P3-08"]
    if task_id in ("P6-01", "P6-02", "P6-03", "P6-04", "P6-05", "P6-07"):
        return ["P2-10"]
    if task_id == "P6-06":
        return ["P2-10", "P6-03"]
    if task_id == "P6-08":
        return ["P2-10", "P5-09"]
    if task_id == "P6-09":
        return ["P2-10", "P5-09", "P6-08"]
    if task_id.startswith("P7-"):
        return ["P5-09", "P6-09"]
    if task_id == "P8-06":
        return ["P7-13"]
    if task_id.startswith("P8-"):
        return []
    return []


DEPENDENCY_RATIONALE = {
    "P3-*": "SPEC 3.3: P3 real 6DoF motion unlocks only after the P2 evidence matrix",
    "P4-*": "SPEC 3.4: real wheel drive needs wheel identity + P2 + P3",
    "P6-static": "SPEC 3.2: P6 static/offline parts need the P2 matrix for the chassis facts",
    "P6-09": "integration manifest needs the arm controller API and the integrated static load",
    "P7-*": "SPEC 3.5: closed-loop hitting needs the arm API and the integrated capability gate",
    "P8-06": "SPEC 3.5: actual PPO training stays gated behind P7-13",
    "P8-design": "SPEC 3.2: P8 design/mock/doc slots are READY now; PPO and the real robot are not authorized",
}


# ------------------------------------------------ proven seeds (never invented)


SEEDS = {
    "P2-01": {
        "status": "PARTIAL",
        "owner": "main",
        "artifact": "outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/G2_r02/result.json",
        "head_sha": "9bf73d7f03b57b4dd0969a331be0d1aa7537bdba",
        "latest_run_id": "G2_r02",
        "next_action": "G2_r02 exists (PASS, exactly-touching zero-g): now re-audit the 'every G0-G4 gate ran twice' wording and land it as an explicit correction",
        "blocker": None,
        "last_error": "repetition claim in the P2-WC report is not yet re-worded; G2_r02 itself passes",
    },
    "P2-02": {
        "status": "PARTIAL",
        "owner": "main",
        "artifact": "outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/G5_r01/manifest.json",
        "head_sha": "6d8fe70981df4fca974ac33f647e8840d9f0eec5",
        "latest_run_id": "G5_r01",
        "next_action": "independently audit the historical G5 manifest (gravity_m_s2 vs --gravity-z -9.81) and add a separate audit file without overwriting the old JSON",
        "blocker": None,
        "last_error": "G5_r01 manifest metadata contradicts the command line; the recorded run is kept as-is",
    },
    "P2-05": {
        "status": "RUNNING",
        "owner": "main",
        "artifact": "management/morph_one_rs/PHASE_P2_WC_G6_COLLISION_IMPLEMENTATIONS.md",
        "head_sha": "9bf73d7f03b57b4dd0969a331be0d1aa7537bdba",
        "latest_run_id": None,
        "next_action": "G6 A/B/C runs already exist (G6A_r01/r02, G6B_r05/r06, G6C_r01/r02): have an independent agent re-audit them before P2-05 may be recorded DONE",
        "blocker": None,
        "last_error": None,
    },
}


def default_entry(task: dict) -> dict:
    return {
        "task_id": task["task_id"],
        "stage": task.get("stage", stage_of(task["task_id"])),
        "title": task.get("title", ""),
        "dependencies": dependencies_for(task["task_id"]),
        "status": "UNKNOWN",
        "owner": None,
        "head_sha": None,
        "artifact": None,
        "latest_run_id": None,
        "next_action": None,
        "blocker": None,
        "last_error": None,
    }


# ------------------------------------------------------------------- state I/O


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=str(path.parent), delete=False
    )
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)


def save_state(state: dict, path: Path) -> None:
    atomic_write_text(path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")


def load_state(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tasks_by_id(state: dict) -> dict:
    return {task["task_id"]: task for task in state.get("tasks", [])}


# ------------------------------------------------------------------- topology


class QueueError(RuntimeError):
    pass


class QueueCycleError(QueueError):
    pass


def topo_order(tasks: dict):
    """Dependency-first order; raises QueueCycleError on any cycle (fail closed)."""
    order = []
    mark = {}

    def visit(node: str, stack):
        state = mark.get(node, 0)
        if state == 2:
            return
        if state == 1:
            cycle = stack[stack.index(node):] + [node]
            raise QueueCycleError("dependency cycle: %s" % " -> ".join(cycle))
        mark[node] = 1
        stack.append(node)
        for dep in tasks[node].get("dependencies", []):
            if dep not in tasks:
                raise QueueError("%s depends on unknown task %s" % (node, dep))
            visit(dep, stack)
        stack.pop()
        mark[node] = 2
        order.append(node)

    for task_id in sorted(tasks):
        visit(task_id, [])
    return order


def blocking_chain(task_id: str, tasks: dict, derived: dict):
    """First non-DONE dependency chain from task_id down to the root cause."""
    chain = [task_id]
    node = task_id
    guard = 0
    while True:
        guard += 1
        if guard > len(tasks) + 1:
            break
        deps = [d for d in tasks[node].get("dependencies", []) if derived.get(d) != DONE]
        if not deps:
            break
        node = deps[0]
        chain.append(node)
    return chain


def compute_queue(state: dict) -> dict:
    """Derive READY/BLOCKED for every slot from the declared statuses."""
    tasks = tasks_by_id(state)
    order = topo_order(tasks)
    derived = {}
    chains = {}
    root_status = {}

    for task_id in order:
        declared = tasks[task_id].get("status", "UNKNOWN")
        if declared not in STATUSES:
            raise QueueError("%s has illegal status %r" % (task_id, declared))
        if declared in (DONE, "RUNNING", "PARTIAL", "FAILED"):
            derived[task_id] = declared
            continue
        pending = [d for d in tasks[task_id].get("dependencies", []) if derived.get(d) != DONE]
        if pending:
            derived[task_id] = "BLOCKED"
            chain = blocking_chain(task_id, tasks, derived)
            chains[task_id] = chain
            root_status[task_id] = derived.get(chain[-1], "UNKNOWN")
        elif declared == "BLOCKED":
            derived[task_id] = "BLOCKED"
            chains[task_id] = [task_id]
            root_status[task_id] = "BLOCKED"
        else:
            derived[task_id] = "READY"

    ready = []
    blocked = []
    counts = {}
    for task_id in order:
        counts[derived[task_id]] = counts.get(derived[task_id], 0) + 1
        entry = {
            "task_id": task_id,
            "stage": tasks[task_id].get("stage", stage_of(task_id)),
            "title": tasks[task_id].get("title", ""),
            "declared_status": tasks[task_id].get("status"),
            "derived_status": derived[task_id],
            "owner": tasks[task_id].get("owner"),
        }
        if derived[task_id] == "READY":
            ready.append(entry)
        elif derived[task_id] == "BLOCKED":
            entry["blocking_chain"] = chains.get(task_id, [task_id])
            entry["root_cause_status"] = root_status.get(task_id)
            blocked.append(entry)

    return {"order": order, "derived": derived, "ready": ready, "blocked": blocked, "counts": counts}


# ----------------------------------------------------------------- generation


def build_registry(spec_tasks, source: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR,
        "source": source,
        "task_count": len(spec_tasks),
        "tasks": [
            {
                "task_id": t["task_id"],
                "stage": t["stage"],
                "title": t["title"],
                "dependencies": dependencies_for(t["task_id"]),
            }
            for t in spec_tasks
        ],
    }


def generate_state(spec_tasks, existing, source: dict) -> dict:
    """Rebuild the queue skeleton, preserving already-recorded runtime facts."""
    old = tasks_by_id(existing) if existing else {}
    tasks = []
    for spec_task in spec_tasks:
        entry = default_entry(spec_task)
        previous = old.get(entry["task_id"])
        if previous:
            for field in (
                "status",
                "owner",
                "head_sha",
                "artifact",
                "latest_run_id",
                "next_action",
                "blocker",
                "last_error",
            ):
                if previous.get(field) is not None:
                    entry[field] = previous[field]
            if previous.get("dependencies"):
                entry["dependencies"] = previous["dependencies"]
        elif entry["task_id"] in SEEDS:
            entry.update(SEEDS[entry["task_id"]])
        tasks.append(entry)

    return {
        "schema_version": SCHEMA_VERSION,
        "project": "Morph One x B601-RS x D455 x Badminton",
        "generator": GENERATOR,
        "spec": {
            "path": SPEC_PATH,
            "ref": source.get("ref"),
            "sha": source.get("sha"),
            "task_count": len(spec_tasks),
        },
        "protocol": "checkpoint-not-stop",
        "whole_project_done": False,
        "status_values": list(STATUSES),
        "dependency_policy": DEPENDENCY_RATIONALE,
        "tasks": tasks,
    }


def validate_state(state: dict):
    """Structural validation; returns a list of human readable problems."""
    problems = []
    if state.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            "schema_version must be %r, found %r" % (SCHEMA_VERSION, state.get("schema_version"))
        )
    tasks = state.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return problems + ["tasks must be a non-empty list"]
    if len(tasks) != EXPECTED_TASK_COUNT:
        problems.append("expected %d task slots, found %d" % (EXPECTED_TASK_COUNT, len(tasks)))

    seen = set()
    known = {t.get("task_id") for t in tasks}
    for task in tasks:
        task_id = task.get("task_id", "<missing>")
        for field in REQUIRED_TASK_FIELDS:
            if field not in task:
                problems.append("%s: missing required field %r" % (task_id, field))
        if task_id in seen:
            problems.append("%s: duplicated task slot" % task_id)
        seen.add(task_id)
        if task.get("status") not in STATUSES:
            problems.append("%s: illegal status %r" % (task_id, task.get("status")))
        deps = task.get("dependencies")
        if not isinstance(deps, list):
            problems.append("%s: dependencies must be a list" % task_id)
            continue
        for dep in deps:
            if dep == task_id:
                problems.append("%s: depends on itself" % task_id)
            elif dep not in known:
                problems.append("%s: unknown dependency %r" % (task_id, dep))
        if task.get("stage") != stage_of(str(task_id)):
            problems.append("%s: stage %r does not match id" % (task_id, task.get("stage")))
        if task.get("status") == DONE:
            if not task.get("artifact"):
                problems.append("%s: DONE without artifact" % task_id)
            if not task.get("head_sha"):
                problems.append("%s: DONE without head_sha" % task_id)
            if not task.get("owner"):
                problems.append("%s: DONE without owner" % task_id)

    try:
        topo_order({t["task_id"]: t for t in tasks})
    except QueueCycleError as exc:
        problems.append(str(exc))
    except (QueueError, KeyError) as exc:
        problems.append("topology error: %s" % exc)

    if state.get("whole_project_done") is True and any(t.get("status") != DONE for t in tasks):
        problems.append("whole_project_done=true while tasks are not DONE")
    return problems


# ------------------------------------------------------------------ transitions

ALLOWED_TRANSITIONS = {
    "UNKNOWN": {"READY", "RUNNING", "DONE", "PARTIAL", "FAILED", "BLOCKED"},
    "READY": {"RUNNING", "DONE", "PARTIAL", "FAILED", "BLOCKED", "UNKNOWN"},
    "RUNNING": {"DONE", "PARTIAL", "FAILED", "BLOCKED"},
    "PARTIAL": {"RUNNING", "DONE", "FAILED", "BLOCKED", "UNKNOWN"},
    "FAILED": {"RUNNING", "PARTIAL", "BLOCKED", "UNKNOWN"},
    "BLOCKED": {"READY", "RUNNING", "UNKNOWN", "PARTIAL", "FAILED"},
    "DONE": set(),
}


def check_transition(task: dict, new_status: str, updates: dict, repo=None):
    """Return the reasons a '--set TASK=STATUS' must be refused (empty = ok)."""
    errors = []
    old_status = task.get("status")
    task_id = task.get("task_id")

    if new_status not in STATUSES:
        return ["illegal status %r (allowed: %s)" % (new_status, ", ".join(STATUSES))]

    allowed = ALLOWED_TRANSITIONS.get(old_status, set())
    if new_status != old_status and new_status not in allowed:
        errors.append("illegal transition %s: %s -> %s" % (task_id, old_status, new_status))
    if new_status == old_status == DONE:
        errors.append("%s: DONE is terminal and its evidence is immutable" % task_id)

    artifact = updates.get("artifact", task.get("artifact"))
    head_sha = updates.get("head_sha", task.get("head_sha"))

    if old_status == DONE and "artifact" in updates and updates["artifact"] != task.get("artifact"):
        errors.append("%s: refusing to replace the evidence of an existing DONE task" % task_id)
    if old_status == DONE and "head_sha" in updates and updates["head_sha"] != task.get("head_sha"):
        errors.append("%s: refusing to replace the head_sha of an existing DONE task" % task_id)

    if new_status == DONE:
        if not artifact:
            errors.append("%s: DONE requires --artifact" % task_id)
        if not head_sha:
            errors.append("%s: DONE requires --head-sha" % task_id)
        if repo is not None and artifact and not artifact_present(repo, artifact, head_sha):
            errors.append("%s: DONE artifact does not exist: %s" % (task_id, artifact))
        if repo is not None and head_sha and not sha_exists(repo, head_sha):
            errors.append("%s: head_sha is not a reachable commit: %s" % (task_id, head_sha))

    if old_status == new_status and new_status != DONE:
        if updates.get("artifact") not in (None, task.get("artifact")) or updates.get(
            "head_sha"
        ) not in (None, task.get("head_sha")):
            errors.append(
                "%s: duplicate write claim: status is already %s with different evidence"
                % (task_id, old_status)
            )
    return errors


def set_status(state: dict, task_id: str, new_status: str, updates: dict, repo=None) -> dict:
    tasks = tasks_by_id(state)
    if task_id not in tasks:
        raise QueueError("unknown task id %r" % task_id)
    task = tasks[task_id]
    errors = check_transition(task, new_status, updates, repo)
    if not errors and new_status in ("READY", DONE):
        derived = compute_queue(state)["derived"]
        pending = [d for d in task.get("dependencies", []) if derived.get(d) != DONE]
        if pending:
            errors.append(
                "%s: cannot be %s, dependencies not DONE: %s"
                % (task_id, new_status, ", ".join(pending))
            )
    if errors:
        raise QueueError("; ".join(errors))

    task["status"] = new_status
    for field in (
        "owner",
        "artifact",
        "head_sha",
        "latest_run_id",
        "next_action",
        "blocker",
        "last_error",
    ):
        if field in updates:
            task[field] = updates[field]
    if new_status == DONE:
        task["blocker"] = None
        task["last_error"] = None
    return task


# ------------------------------------------------------------------------- CLI


def _reconfigure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _parse_updates(args) -> dict:
    updates = {}
    for field, value in (
        ("owner", args.owner),
        ("artifact", args.artifact),
        ("head_sha", args.head_sha),
        ("latest_run_id", args.run_id),
        ("next_action", args.next_action),
        ("blocker", args.blocker),
        ("last_error", args.last_error),
    ):
        if value is not None:
            updates[field] = value
    return updates


def cmd_generate(args, repo: Path, state_path: Path) -> int:
    try:
        text, source = read_spec(repo, args.spec_ref, args.spec_path)
        spec_tasks = parse_spec(text)
    except (SpecError, RuntimeError) as exc:
        registry_path = repo / args.registry
        if not registry_path.is_file():
            print("error: %s" % exc, file=sys.stderr)
            return 2
        registry = load_state(registry_path)
        spec_tasks = registry["tasks"]
        source = registry.get("source", {})
        print("note: spec unavailable (%s); using committed registry %s" % (exc, args.registry),
              file=sys.stderr)

    existing = None
    if state_path.is_file() and not args.reset:
        existing = load_state(state_path)
    state = generate_state(spec_tasks, existing, source)

    registry = build_registry(spec_tasks, source)
    atomic_write_text(repo / args.registry, json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    save_state(state, state_path)
    derived = compute_queue(state)
    shown = state_path.as_posix()
    if shown.startswith(repo.as_posix() + "/"):
        shown = shown[len(repo.as_posix()) + 1:]
    print(
        "generated %s: %d slots (%s)"
        % (shown, len(state["tasks"]),
           ", ".join("%s=%d" % (k, derived["counts"][k]) for k in sorted(derived["counts"])))
    )
    print("registry: %s" % args.registry)
    return 0


def cmd_status(args, repo: Path, state_path: Path) -> int:
    state = load_state(state_path)
    queue = compute_queue(state)
    tasks = tasks_by_id(state)
    if args.json:
        print(json.dumps({
            "schema_version": state.get("schema_version"),
            "counts": queue["counts"],
            "tasks": [
                {
                    "task_id": tid,
                    "stage": tasks[tid].get("stage"),
                    "declared_status": tasks[tid].get("status"),
                    "derived_status": queue["derived"][tid],
                    "owner": tasks[tid].get("owner"),
                    "artifact": tasks[tid].get("artifact"),
                }
                for tid in queue["order"]
            ],
        }, ensure_ascii=False, indent=2))
        return 0
    print("%-8s %-9s %-9s %-10s %s" % ("TASK", "DECLARED", "DERIVED", "OWNER", "ARTIFACT"))
    for task_id in queue["order"]:
        task = tasks[task_id]
        print("%-8s %-9s %-9s %-10s %s" % (
            task_id,
            task.get("status"),
            queue["derived"][task_id],
            task.get("owner") or "-",
            task.get("artifact") or "-",
        ))
    print("")
    print("counts: %s" % json.dumps(queue["counts"], ensure_ascii=False, sort_keys=True))
    return 0


def cmd_ready(args, repo: Path, state_path: Path) -> int:
    state = load_state(state_path)
    queue = compute_queue(state)
    payload = {
        "schema_version": state.get("schema_version"),
        "rule": "READY <=> every dependency is DONE",
        "counts": queue["counts"],
        "ready": queue["ready"],
        "blocked": queue["blocked"],
    }
    if args.stage:
        stages = {s.strip() for s in args.stage.split(",") if s.strip()}
        payload["ready"] = [t for t in payload["ready"] if t["stage"] in stages]
    payload["count"] = len(payload["ready"])
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_set(args, repo: Path, state_path: Path) -> int:
    pattern = r"[A-Za-z0-9_-]+=(%s)" % "|".join(STATUSES)
    if not re.fullmatch(pattern, args.set):
        print("error: --set expects TASK_ID=STATUS (status in %s)" % ", ".join(STATUSES),
              file=sys.stderr)
        return 2
    task_id, new_status = args.set.split("=", 1)
    state = load_state(state_path)
    try:
        task = set_status(state, task_id, new_status, _parse_updates(args), repo)
    except QueueError as exc:
        print("refused: %s" % exc, file=sys.stderr)
        return 1
    save_state(state, state_path)
    print("%s -> %s (artifact=%s, head_sha=%s)"
          % (task_id, new_status, task.get("artifact"), task.get("head_sha")))
    return 0


def cmd_validate(args, repo: Path, state_path: Path) -> int:
    state = load_state(state_path)
    problems = validate_state(state)
    try:
        compute_queue(state)
    except QueueCycleError as exc:
        problems.append(str(exc))
    if problems:
        for problem in problems:
            print("INVALID %s" % problem, file=sys.stderr)
        return 1
    print("state OK: %d slots, schema %s" % (len(state["tasks"]), state.get("schema_version")))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Queue state + dependency-aware READY selector (pure CPU, stdlib only)."
    )
    parser.add_argument("--state", default=STATE_PATH, help="queue state file (repo relative)")
    parser.add_argument("--registry", default=REGISTRY_PATH, help="task registry snapshot")
    parser.add_argument("--spec-path", default=SPEC_PATH, help="queue spec markdown path")
    parser.add_argument("--spec-ref", default=None, help="git ref holding the spec if absent locally")
    parser.add_argument("--repo", default=None, help="repository root override")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true", help="print the queue table")
    group.add_argument("--ready", action="store_true", help="print ready slots as JSON")
    group.add_argument("--set", metavar="TASK_ID=STATUS", help="record a status transition")
    group.add_argument("--validate", action="store_true", help="structural validation")
    group.add_argument("--generate", action="store_true", help="(re)generate state + registry")
    parser.add_argument("--stage", default=None, help="comma separated stage filter for --ready")
    parser.add_argument("--json", action="store_true", help="machine readable --status")
    parser.add_argument("--reset", action="store_true", help="with --generate: drop recorded facts")
    parser.add_argument("--owner")
    parser.add_argument("--artifact")
    parser.add_argument("--head-sha", dest="head_sha")
    parser.add_argument("--run-id", dest="run_id")
    parser.add_argument("--next-action", dest="next_action")
    parser.add_argument("--blocker")
    parser.add_argument("--last-error", dest="last_error")
    return parser


def main(argv=None) -> int:
    _reconfigure_stdout()
    args = build_parser().parse_args(argv)
    repo = repo_root(args.repo)
    state_path = repo / args.state
    try:
        if args.generate:
            return cmd_generate(args, repo, state_path)
        if not state_path.is_file():
            print("error: queue state not found: %s" % state_path, file=sys.stderr)
            return 2
        if args.status:
            return cmd_status(args, repo, state_path)
        if args.ready:
            return cmd_ready(args, repo, state_path)
        if args.set:
            return cmd_set(args, repo, state_path)
        if args.validate:
            return cmd_validate(args, repo, state_path)
    except QueueCycleError as exc:
        print("FAIL CLOSED: %s" % exc, file=sys.stderr)
        return 3
    except QueueError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
