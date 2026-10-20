"""Attribute the differences between two STEP -> USD mappings to their causes.

Why this exists
---------------
The audited mapper changed two things at once: the containment inequality inside
the bbox sandwich, and the pairing algorithm (greedy per name-key group became a
maximum-cardinality augmenting-path matching).  A reviewer needs to know which
change is responsible for each moved part, so this tool separates them.

It never edits the audited mapper.  It imports that mapper only for the STEP/USD
loaders, and re-implements *both* containment rules here:

  pre-audit rule   below = max(0, lower_min - mesh_min, lower_max - mesh_max)
  audited rule     below = max(0, mesh_min - lower_min, lower_max - mesh_max)

The pre-audit rule is reproduced so the old mapping can be validated against it;
if the reproduction does not reproduce the committed old result, every
attribution below would be suspect and the report says so.

Cause labels
  INEQUALITY_FIX   the new partner was not admissible under the pre-audit rule
  MATCHING_ALGORITHM  both partners were admissible; the augmenting-path matching
                   chose a place the greedy pass did not
  UNCHANGED        same solid, same mesh
  NEWLY_MATCHED / LOST  appeared or disappeared relative to the old mapping
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

__all__ = ["old_rule_ok", "audited_rule_ok", "group_feasibility", "compare"]

TOL_MM = 2.5
TOL_REL = 0.01


def _tol(extent: float) -> float:
    return max(abs(extent) * TOL_REL, TOL_MM)


def _bounds(lower_min, lower_max, upper_min, upper_max, mesh_min, mesh_max):
    return (lower_min, lower_max, upper_min, upper_max, mesh_min, mesh_max)


def _violations(lower_min, lower_max, upper_min, upper_max, mesh_min, mesh_max, mode: str):
    below = []
    above = []
    for axis in range(3):
        extent = max(
            abs(lower_max[axis] - lower_min[axis]),
            abs(upper_max[axis] - upper_min[axis]),
        )
        tolerance = _tol(extent)
        if mode == "old":
            # pre-audit: penalised a mesh that reached below the vertex box and
            # failed to notice a mesh that did not cover the vertex box at all
            raw_below = max(
                0.0,
                lower_min[axis] - mesh_min[axis],
                lower_max[axis] - mesh_max[axis],
            )
        else:
            raw_below = max(
                0.0,
                mesh_min[axis] - lower_min[axis],
                lower_max[axis] - mesh_max[axis],
            )
        raw_above = max(
            0.0,
            upper_min[axis] - mesh_min[axis],
            mesh_max[axis] - upper_max[axis],
        )
        below.append(raw_below if raw_below > tolerance else 0.0)
        above.append(raw_above if raw_above > tolerance else 0.0)
    return below, above


def old_rule_ok(solid, mesh) -> bool:
    below, above = _violations(
        solid["lower_min"], solid["lower_max"], solid["upper_min"], solid["upper_max"],
        mesh["min"], mesh["max"], "old",
    )
    return all(v == 0.0 for v in below) and all(v == 0.0 for v in above)


def audited_rule_ok(solid, mesh) -> bool:
    below, above = _violations(
        solid["lower_min"], solid["lower_max"], solid["upper_min"], solid["upper_max"],
        mesh["min"], mesh["max"], "audited",
    )
    return all(v == 0.0 for v in below) and all(v == 0.0 for v in above)


def _cost(solid, mesh) -> float:
    total = 0.0
    for axis in range(3):
        extent = max(
            abs(solid["lower_max"][axis] - solid["lower_min"][axis]),
            abs(solid["upper_max"][axis] - solid["upper_min"][axis]),
            1.0,
        )
        scale = max(0.02 * extent, 1.0)
        total += abs(mesh["min"][axis] - solid["lower_min"][axis]) / scale
        total += abs(mesh["max"][axis] - solid["lower_max"][axis]) / scale * 0.5
    return total


def group_feasibility(solids: Sequence[dict], meshes: Sequence[dict], ok_fn) -> Dict[int, set]:
    """solid eid -> set of mesh paths admissible under one containment rule."""
    out: Dict[int, set] = defaultdict(set)
    for solid in solids:
        for mesh in meshes:
            if ok_fn(solid, mesh):
                out[solid["eid"]].add(mesh["path"])
    return out


def _greedy_old(solids: Sequence[dict], meshes: Sequence[dict]) -> Dict[int, str]:
    """Reproduce the pre-audit selection: per name-key group, cheapest feasible first."""
    from scripts.simulation.map_step_solids_to_usd_meshes import name_key

    by_key_solids: Dict[str, List[dict]] = defaultdict(list)
    by_key_meshes: Dict[str, List[dict]] = defaultdict(list)
    for solid in solids:
        by_key_solids[name_key(solid["name"])].append(solid)
    for mesh in meshes:
        by_key_meshes[name_key(mesh["core"])].append(mesh)

    chosen: Dict[int, str] = {}
    for key, group_solids in by_key_solids.items():
        group_meshes = by_key_meshes.get(key, [])
        scored = []
        for solid in group_solids:
            for mesh in group_meshes:
                if old_rule_ok(solid, mesh):
                    scored.append((_cost(solid, mesh), solid["eid"], mesh["path"]))
        scored.sort()
        taken_solids: set = set()
        taken_meshes: set = set()
        for _, eid, path in scored:
            if eid in taken_solids or path in taken_meshes:
                continue
            chosen[eid] = path
            taken_solids.add(eid)
            taken_meshes.add(path)
    return chosen


def compare(solids, meshes, old_pairs, new_pairs, reproduced_old) -> dict:
    old_by_eid = {int(pair["step_eid"]): pair for pair in old_pairs}
    new_by_eid = {int(pair["step_eid"]): pair for pair in new_pairs}

    old_feasible = group_feasibility(solids, meshes, old_rule_ok)
    audited_feasible = group_feasibility(solids, meshes, audited_rule_ok)
    all_mesh_paths = {mesh["path"] for mesh in meshes}

    reproduced_agreement = 0
    for eid, path in reproduced_old.items():
        pair = old_by_eid.get(eid)
        if pair is not None and pair["usd_path"] == path:
            reproduced_agreement += 1

    rows = []
    causes: Counter = Counter()
    for eid in sorted(set(old_by_eid) | set(new_by_eid)):
        old_pair = old_by_eid.get(eid)
        new_pair = new_by_eid.get(eid)
        old_path = old_pair["usd_path"] if old_pair else None
        new_path = new_pair["usd_path"] if new_pair else None
        if old_path == new_path:
            cause = "UNCHANGED"
        elif old_pair is None:
            # The solid had no partner before.  If its new partner was already
            # admissible under the pre-audit rule, the old greedy pass simply
            # failed to place it (coverage), which is a matching effect.
            cause = (
                "MATCHING_ALGORITHM"
                if new_path in old_feasible.get(eid, set())
                else "INEQUALITY_FIX"
            )
        elif new_pair is None:
            if old_path not in all_mesh_paths:
                cause = "LOST"  # the mesh itself is gone; no rule explains that
            elif old_path not in audited_feasible.get(eid, set()):
                cause = "INEQUALITY_FIX"
            else:
                cause = "MATCHING_ALGORITHM"
        else:
            new_ok_under_old = new_path in old_feasible.get(eid, set())
            old_ok_under_audited = old_path in audited_feasible.get(eid, set())
            if not new_ok_under_old:
                cause = "INEQUALITY_FIX"
            elif not old_ok_under_audited:
                cause = "INEQUALITY_FIX"
            else:
                cause = "MATCHING_ALGORITHM"
        causes[cause] += 1
        rows.append(
            {
                "step_eid": eid,
                "step_name": (new_pair or old_pair)["step_name"],
                "old_usd_path": old_path,
                "new_usd_path": new_path,
                "old_grade": old_pair["grade"] if old_pair else None,
                "new_grade": new_pair["grade"] if new_pair else None,
                "cause": cause,
                "new_partner_feasible_under_old_rule": (
                    new_path in old_feasible.get(eid, set()) if new_path else None
                ),
                "old_partner_feasible_under_audited_rule": (
                    old_path in audited_feasible.get(eid, set()) if old_path else None
                ),
                "candidate_count_old_rule": len(old_feasible.get(eid, set())),
                "candidate_count_audited_rule": len(audited_feasible.get(eid, set())),
            }
        )

    return {
        "old_pairs": len(old_pairs),
        "new_pairs": len(new_pairs),
        "cause_counts": dict(causes),
        "reproduced_old_rule_pairs": len(reproduced_old),
        "reproduced_old_rule_matches_committed_old": reproduced_agreement,
        "reproduced_old_rule_agreement_ratio": round(
            reproduced_agreement / max(1, min(len(reproduced_old), len(old_pairs))), 4
        ),
        "tolerance": {"abs_mm": TOL_MM, "rel": TOL_REL},
        "rows": rows,
    }


# ------------------------------------------------------------------------ CLI


def main(argv: "Sequence[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explain what moved between two STEP->USD mappings"
    )
    parser.add_argument("--step", required=True)
    parser.add_argument("--usd", required=True)
    parser.add_argument("--old-report", required=True, help="mapping JSON from the pre-audit run")
    parser.add_argument("--new-report", required=True, help="mapping JSON from the audited run")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    from scripts.simulation.map_step_solids_to_usd_meshes import (
        load_mesh_facts,
        load_solid_facts,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    solids = load_solid_facts(args.step)
    meshes = load_mesh_facts(args.usd)
    old_pairs = json.loads(Path(args.old_report).read_text(encoding="utf-8"))["pairs"]
    new_pairs = json.loads(Path(args.new_report).read_text(encoding="utf-8"))["pairs"]
    reproduced = _greedy_old(solids, meshes)

    result = compare(solids, meshes, old_pairs, new_pairs, reproduced)
    result["inputs"] = {
        "step": str(Path(args.step).resolve()),
        "usd": str(Path(args.usd).resolve()),
        "old_report": str(Path(args.old_report).resolve()),
        "new_report": str(Path(args.new_report).resolve()),
    }
    result["grade_change_counts"] = dict(
        Counter(
            "%s -> %s" % (row["old_grade"], row["new_grade"]) for row in result["rows"]
        )
    )

    json_path = out_dir / "mapping_supersession.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    csv_path = out_dir / "mapping_supersession.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cause", "step_eid", "old_grade", "new_grade", "old_usd_path", "new_usd_path",
                "candidates_old_rule", "candidates_audited_rule",
                "new_partner_ok_old_rule", "old_partner_ok_audited_rule", "step_name",
            ]
        )
        for row in result["rows"]:
            writer.writerow(
                [
                    row["cause"], row["step_eid"], row["old_grade"], row["new_grade"],
                    row["old_usd_path"], row["new_usd_path"],
                    row["candidate_count_old_rule"], row["candidate_count_audited_rule"],
                    row["new_partner_feasible_under_old_rule"],
                    row["old_partner_feasible_under_audited_rule"], row["step_name"],
                ]
            )

    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "old_pairs": result["old_pairs"],
                "new_pairs": result["new_pairs"],
                "cause_counts": result["cause_counts"],
                "grade_change_counts": result["grade_change_counts"],
                "reproduced_old_rule_pairs": result["reproduced_old_rule_pairs"],
                "reproduced_old_rule_agreement_ratio": result[
                    "reproduced_old_rule_agreement_ratio"
                ],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
