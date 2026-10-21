"""Offline contract checks for the 96-item resumable engineering queue."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "simulation"))
from task_queue import read_queue, ready_tasks


class ContinuousQueueTests(unittest.TestCase):
    def test_96_unique_items_and_safe_statuses(self):
        data = read_queue()
        self.assertEqual(len(data["tasks"]), 96)
        self.assertEqual(len({t["task_id"] for t in data["tasks"]}), 96)

    def test_g5_followup_ready_and_p3_blocked(self):
        tasks = {t["task_id"]: t for t in read_queue()["tasks"]}
        self.assertEqual(tasks["P2-04"]["status"], "READY")
        self.assertEqual(tasks["P3-01"]["status"], "BLOCKED")
        self.assertIn("P2_FINAL_PASS", tasks["P3-01"]["dependencies"])

    def test_ppo_execution_not_authorized(self):
        data = read_queue()
        self.assertFalse(data["autonomous_execution"])
        tasks = {t["task_id"]: t for t in data["tasks"]}
        self.assertEqual(tasks["P8-06"]["status"], "BLOCKED")
        self.assertIn("USER_PPO_AUTHORIZATION", tasks["P8-06"]["dependencies"])

    def test_ready_does_not_include_blocked(self):
        data = read_queue()
        ready = ready_tasks(data, "gpu", 15)
        self.assertTrue(ready)
        self.assertTrue(all(t["resource"] == "gpu" and t["status"] == "READY"
                            for t in ready))

    def test_duplicate_ids_rejected(self):
        data = read_queue()
        data["tasks"].append(dict(data["tasks"][0]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queue.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_queue(path)

    def test_ready_with_dependency_rejected(self):
        data = read_queue()
        task = data["tasks"][0]
        task["dependencies"] = ["P2_FINAL_PASS"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queue.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "uncleared"):
                read_queue(path)

    def test_done_without_evidence_rejected(self):
        data = read_queue()
        data["tasks"][0]["status"] = "DONE"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queue.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "without evidence"):
                read_queue(path)


if __name__ == "__main__":
    unittest.main()
