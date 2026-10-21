"""Validate and inspect the Morph One P0-P8 queue. Does NOT run tasks.

Usage:
 python scripts/simulation/task_queue.py --validate
 python scripts/simulation/task_queue.py --resource gpu --limit 12
 python scripts/simulation/task_queue.py --resource cpu --limit 12
"""
import argparse
import json
from pathlib import Path

STATUSES = {"READY", "RUNNING", "DONE", "PARTIAL", "FAILED", "BLOCKED", "UNKNOWN"}
RESOURCES = {"cpu", "gpu", "design"}
DEFAULT = Path(__file__).resolve().parents[2] / "management" / "morph_one_rs" / "EXECUTION_QUEUE_STATE.json"


def read_queue(path=DEFAULT):
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("queue tasks must be a nonempty array")
    ids = [task.get("task_id") for task in tasks]
    if any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ValueError("invalid task id")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task IDs")
    for task in tasks:
        if task.get("status") not in STATUSES:
            raise ValueError("bad task status: " + str(task.get("task_id")))
        if task.get("resource") not in RESOURCES:
            raise ValueError("bad resource: " + str(task.get("task_id")))
        if not isinstance(task.get("dependencies"), list):
            raise ValueError("dependencies must be a list")
        if task.get("status") == "READY" and task.get("dependencies"):
            # This queue never auto-unlocks a dependency. An explicitly revised
            # gate must be audited and manually marked DONE before marking READY.
            raise ValueError("READY with uncleared dependencies: " + task["task_id"])
        if task.get("status") == "DONE" and not task.get("evidence_path"):
            raise ValueError("DONE without evidence path: " + task["task_id"])
    return data


def ready_tasks(data, resource=None, limit=None):
    tasks = [task for task in data["tasks"] if task["status"] == "READY"]
    if resource is not None:
        tasks = [t for t in tasks if t["resource"] == resource]
    tasks.sort(key=lambda t: (t.get("priority", 999), t["task_id"]))
    return tasks if limit is None else tasks[:limit]


def main():
    parser = argparse.ArgumentParser(description="Morph One P0-P8 resumable queue inspector")
    parser.add_argument("--file", type=Path, default=DEFAULT)
    parser.add_argument("--resource", choices=sorted(RESOURCES))
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must not be negative")
    data = read_queue(args.file)
    all_tasks = data["tasks"]
    counts = {status: sum(t["status"] == status for t in all_tasks) for status in sorted(STATUSES)}
    if args.validate:
        print(json.dumps({"valid": True, "total": len(all_tasks), "statuses": counts}, ensure_ascii=False))
    else:
        print(json.dumps({"total": len(all_tasks), "statuses": counts, "ready": [
            {"task_id": task["task_id"], "resource": task["resource"],
             "description": task["description"]}
            for task in ready_tasks(data, args.resource, args.limit)
        ]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
