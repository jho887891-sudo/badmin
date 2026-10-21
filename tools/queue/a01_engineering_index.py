#!/usr/bin/env python
"""A01: machine-readable engineering index for the Morph One / Badmin repository.

Everything in management/morph_one_rs/ENGINEERING_INDEX.json is produced by this
script: branches -> HEAD sha, worktrees, pull requests, key documents and key
evidence directories. Nothing is hand typed.

Pure CPU, python standard library only (git and gh are called as external tools
and their absence degrades to an explicit "unavailable" record). Never imports
Isaac/Kit/omni/pxr/torch and never launches a simulator.

    python tools/queue/a01_engineering_index.py            # write the index
    python tools/queue/a01_engineering_index.py --check     # fail if stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "1.0"
GENERATOR = "tools/queue/a01_engineering_index.py"
OUT_PATH = "management/morph_one_rs/ENGINEERING_INDEX.json"
SPEC_PATH = "management/morph_one_rs/EXECUTION_QUEUE_CONTINUOUS_P0_P8.md"
SPEC_REF = "origin/chatgpt/p2-wc-continuous-queue-20261021"
EVIDENCE_ROOT = "outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates"
DOC_ROOTS = ("management", "docs", "deliverables")
EVIDENCE_MAX_DIRS = 400
# when the evidence tree is not in this worktree, read it from these refs instead
EVIDENCE_REFS = ("origin/dev/p2-wheel-contact-g1g4",)


def repo_root(start=None) -> Path:
    if start is not None:
        return Path(start).resolve()
    return Path(__file__).resolve().parents[2]


def run(cmd, cwd, timeout=60):
    """Return (exit_code, stdout) and never raise: external tools may be absent."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, "%s: %s" % (type(exc).__name__, exc)
    if proc.returncode != 0:
        return proc.returncode, proc.stderr.strip()
    return 0, proc.stdout


def git(repo: Path, *args: str):
    return run(["git", "-C", str(repo), *args], repo)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------- branches


def collect_branches(repo: Path):
    code, out = git(
        repo,
        "for-each-ref",
        "--format=%(refname:short)\t%(objectname)\t%(committerdate:iso-strict)\t%(contents:subject)",
        "refs/remotes/origin",
        "refs/heads",
    )
    branches = []
    if code != 0:
        return branches, "git for-each-ref failed: %s" % out
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        name, sha, date, subject = parts
        if name.endswith("/HEAD"):
            continue
        branches.append(
            {"name": name, "head_sha": sha, "committed_at": date, "subject": subject}
        )
    branches.sort(key=lambda item: item["name"])
    return branches, None


def collect_worktrees(repo: Path):
    code, out = git(repo, "worktree", "list", "--porcelain")
    worktrees = []
    if code != 0:
        return worktrees
    current = {}
    for line in out.splitlines() + [""]:
        if not line:
            if current:
                worktrees.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            current["path"] = value
        elif key == "HEAD":
            current["head_sha"] = value
        elif key == "branch":
            current["branch"] = value.replace("refs/heads/", "")
        elif key == "detached":
            current["detached"] = True
        elif key == "bare":
            current["bare"] = True
    worktrees.sort(key=lambda item: item.get("path", ""))
    return worktrees


def collect_pulls(repo: Path):
    code, out = run(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "all",
            "--limit",
            "200",
            "--json",
            "number,title,state,isDraft,headRefName,baseRefName,updatedAt,url",
        ],
        repo,
        timeout=90,
    )
    if code != 0:
        return [], "gh unavailable or failed: %s" % out
    try:
        pulls = json.loads(out)
    except ValueError as exc:
        return [], "gh returned invalid JSON: %s" % exc
    for pull in pulls:
        pull["number"] = int(pull.get("number", 0))
    pulls.sort(key=lambda item: item["number"])
    return pulls, None


def collect_branch_protection(repo: Path, default_branch):
    if not default_branch:
        return {"available": False, "reason": "unknown default branch"}
    code, out = run(
        ["gh", "api", "repos/{owner}/{repo}/branches/%s/protection" % default_branch],
        repo,
        timeout=60,
    )
    if code != 0:
        return {"available": False, "branch": default_branch, "reason": out.strip()[:300]}
    try:
        return {"available": True, "branch": default_branch, "detail": json.loads(out)}
    except ValueError:
        return {"available": False, "branch": default_branch, "reason": "invalid JSON"}


# ------------------------------------------------------- documents / evidence


def collect_documents(repo: Path):
    documents = []
    seen = set()
    for root_name in DOC_ROOTS:
        root = repo / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.md")):
            rel = path.relative_to(repo).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            documents.append(
                {
                    "path": rel,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    documents.sort(key=lambda item: item["path"])
    return documents


def collect_evidence(repo: Path):
    root = repo / EVIDENCE_ROOT
    runs = []
    totals = {"run_dirs": 0, "files": 0, "bytes": 0}
    if not root.is_dir():
        return {"root": EVIDENCE_ROOT, "runs": [], "totals": totals}
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(p.name for p in run_dir.iterdir() if p.is_file())
        size = sum((run_dir / name).stat().st_size for name in files)
        manifest = run_dir / "manifest.json"
        result = run_dir / "result.json"
        runs.append(
            {
                "run_id": run_dir.name,
                "path": run_dir.relative_to(repo).as_posix(),
                "files": files,
                "bytes": size,
                "has_manifest": manifest.is_file(),
                "has_result": result.is_file(),
                "has_raw_poses": (run_dir / "raw_poses.jsonl").is_file(),
            }
        )
        totals["run_dirs"] += 1
        totals["files"] += len(files)
        totals["bytes"] += size
    runs = runs[:EVIDENCE_MAX_DIRS]
    return {"root": EVIDENCE_ROOT, "runs": runs, "totals": totals}


def collect_evidence_refs(repo: Path):
    """Run dirs listed from git refs, for worktrees that do not hold the tree."""
    collected = []
    for ref in EVIDENCE_REFS:
        code, out = git(repo, "ls-tree", "-r", "--name-only", ref, "--", EVIDENCE_ROOT)
        if code != 0:
            continue
        runs = {}
        for name in out.splitlines():
            name = name.strip()
            if not name:
                continue
            rel = name[len(EVIDENCE_ROOT) + 1:]
            if "/" not in rel:
                continue
            run_id, _, filename = rel.partition("/")
            runs.setdefault(run_id, []).append(filename)
        for run_id in sorted(runs):
            files = sorted(runs[run_id])
            collected.append(
                {
                    "run_id": run_id,
                    "path": "%s/%s" % (EVIDENCE_ROOT, run_id),
                    "source": "git:%s" % ref,
                    "files": files,
                    "has_manifest": "manifest.json" in files,
                    "has_result": "result.json" in files,
                    "has_raw_poses": "raw_poses.jsonl" in files,
                }
            )
    collected.sort(key=lambda item: (item["path"], item.get("source", "")))
    return collected


def collect_outputs(repo: Path):
    root = repo / "outputs"
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        files = sum(1 for _ in path.rglob("*") if _.is_file())
        out.append({"path": path.relative_to(repo).as_posix(), "files": files})
    return out


# ---------------------------------------------------------------------- index


def build_index(repo: Path) -> dict:
    branches, branches_error = collect_branches(repo)
    worktrees = collect_worktrees(repo)
    pulls, pulls_error = collect_pulls(repo)

    code, head_sha = git(repo, "rev-parse", "HEAD")
    head_sha = head_sha.strip() if code == 0 else None
    code, head_branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    head_branch = head_branch.strip() if code == 0 else None
    code, default_ref = git(repo, "symbolic-ref", "refs/remotes/origin/HEAD")
    default_branch = None
    if code == 0:
        default_branch = default_ref.strip().split("/")[-1]

    code, unmerged = git(repo, "branch", "-r", "--no-merged", "origin/main", "--format=%(refname:short)")
    unmerged_branches = sorted(l.strip() for l in unmerged.splitlines() if l.strip()) if code == 0 else []

    merged = set()
    code, merged_out = git(repo, "branch", "-r", "--merged", "origin/main", "--format=%(refname:short)")
    if code == 0:
        merged = {line.strip() for line in merged_out.splitlines() if line.strip()}

    pr_by_branch = {}
    for pull in pulls:
        pr_by_branch.setdefault(pull.get("headRefName"), []).append(pull.get("number"))
    for branch in branches:
        short = branch["name"]
        branch["is_default"] = short == "origin/%s" % default_branch if default_branch else False
        branch["merged_into_main"] = short in merged
        branch["pull_requests"] = sorted(pr_by_branch.get(short.replace("origin/", ""), []))

    code, spec_sha = git(repo, "rev-parse", SPEC_REF)
    spec_sha = spec_sha.strip() if code == 0 else None
    code, spec_blob = git(repo, "rev-parse", "%s:%s" % (SPEC_REF, SPEC_PATH))
    spec_blob = spec_blob.strip() if code == 0 else None

    notes = []
    if default_branch is None:
        notes.append(
            "origin/HEAD is not set: main is NOT proven to be the default branch, "
            "so no branch is marked is_default here"
        )
    if branches_error:
        notes.append(branches_error)
    if pulls_error:
        notes.append(pulls_error)

    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": GENERATOR,
        "repo": {
            "root": ".",  # deliberately relative: the index must stay worktree independent
            "head_sha": head_sha,
            "head_branch": head_branch,
            "default_branch": default_branch,
            "default_branch_source": "git symbolic-ref refs/remotes/origin/HEAD",
        },
        "spec": {
            "path": SPEC_PATH,
            "ref": SPEC_REF,
            "ref_sha": spec_sha,
            "blob_sha": spec_blob,
            "read_by": "tools/queue/queue_state.py --generate",
        },
        "branch_count": len(branches),
        "branches": branches,
        "unmerged_branches": unmerged_branches,
        "worktree_count": len(worktrees),
        "worktrees": worktrees,
        "pull_count": len(pulls),
        "pulls": pulls,
        "branch_protection": collect_branch_protection(repo, default_branch),
        "documents": collect_documents(repo),
        "evidence": collect_evidence(repo),
        "evidence_from_refs": collect_evidence_refs(repo),
        "outputs_roots": collect_outputs(repo),
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "pure_cpu": True,
            "launches_isaac_or_kit": False,
        },
        "notes": notes,
    }
    return index


# Ref-shaped facts move whenever any agent pushes. They are recorded in the
# snapshot but excluded from the freshness comparison, otherwise --check would
# fail purely because a colleague pushed a branch while we were reading.
MOVING_KEYS = (
    "generated_at_utc",
    "branches",
    "branch_count",
    "unmerged_branches",
    "pulls",
    "pull_count",
    "worktrees",
    "worktree_count",
    "branch_protection",
    "evidence_from_refs",
)


def strip_volatile(index: dict) -> dict:
    """Stable facts only: schema, spec, documents, evidence, outputs, environment."""
    clone = json.loads(json.dumps(index))
    for key in MOVING_KEYS:
        clone.pop(key, None)
    env = clone.get("environment")
    if isinstance(env, dict):
        env.pop("platform", None)
    return clone


def strip_moving(index: dict) -> dict:
    """Ref-shaped facts only (branch/PR/worktree/run listings)."""
    clone = json.loads(json.dumps(index))
    keep = {key: clone.get(key) for key in MOVING_KEYS if key != "generated_at_utc"}
    return keep


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="A01 engineering index generator (pure CPU).")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--check", action="store_true", help="fail when the committed index is stale")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="with --check: also fail when branch/PR/worktree snapshots drifted",
    )
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    repo = repo_root(args.repo)
    out_path = repo / args.out
    index = build_index(repo)
    text = json.dumps(index, ensure_ascii=False, indent=2) + "\n"

    if args.check:
        if not out_path.is_file():
            print("STALE: %s is missing" % args.out, file=sys.stderr)
            return 1
        current = json.loads(out_path.read_text(encoding="utf-8"))
        if strip_volatile(current) != strip_volatile(index):
            print("STALE: %s stable facts (spec/documents/evidence) do not match a fresh scan; "
                  "rerun without --check" % args.out, file=sys.stderr)
            return 1
        drifted = [
            key for key in strip_moving(current)
            if strip_moving(current)[key] != strip_moving(index)[key]
        ]
        if drifted and args.strict:
            print("STALE (strict): moving refs drifted: %s" % ", ".join(drifted), file=sys.stderr)
            return 1
        print("index OK: %s matches a fresh scan (stable facts)" % args.out)
        if drifted:
            print("  note: moving refs changed since the snapshot: %s" % ", ".join(drifted))
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8", newline="\n")
    print(
        "wrote %s: %d branches, %d worktrees, %d pulls, %d documents, %d run dirs"
        % (
            args.out,
            index["branch_count"],
            index["worktree_count"],
            index["pull_count"],
            len(index["documents"]),
            index["evidence"]["totals"]["run_dirs"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
