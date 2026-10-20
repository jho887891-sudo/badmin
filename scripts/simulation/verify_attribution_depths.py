"""Independent check of the candidate-outlier -> solid attribution depth structure.

Why this exists
---------------
The pre-audit attribution stopped walking upward as soon as the first owning
solid was found, which can miss a second owner reachable only through a longer
branch.  The audit removed that early stop.  Re-running the tool produced
byte-identical output on the real chassis file, so the interesting question is
*why* nothing changed -- "the hashes match" is not an explanation.

This script answers that structurally: for every candidate point it walks the
reverse reference graph level by level and records the solids found at each
depth.  Then:

* legacy_owners  = solids found at the first depth that yields any owner
                   (exactly what the old early-stop could ever see),
* full_owners    = union over all depths (what the audit now returns).

If legacy == full for every point, the early stop was harmless for this model,
and the report says so with numbers instead of hope.  Points where the two sets
differ are listed with their entity ids and the full reference chains, which is
what a reviewer needs to check a real behavioural change.

The script never edits the census module: it imports the audited implementation
for parsing and closure, and reproduces only the *stopping rule* difference here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from scripts.simulation.step_geometry_census import (
    SOLID_TYPES,
    Entity,
    classify_point,
    entity_name,
    parse_step,
    point_statistics,
    reverse_index,
    walk_up_paths,
)

__all__ = ["owners_by_level", "verify"]


def owners_by_level(
    entities: Dict[int, Entity],
    reverse: Dict[int, List[int]],
    point_id: int,
    max_hops: int = 40,
) -> List[Tuple[int, List[int]]]:
    """Solids reachable from a point, grouped by the depth at which they appear."""
    visited: Set[int] = {point_id}
    frontier = [point_id]
    levels: List[Tuple[int, List[int]]] = []
    for depth in range(1, max_hops + 1):
        nxt: List[int] = []
        owners: Set[int] = set()
        for eid in frontier:
            for parent in reverse.get(eid, ()):  # type: ignore[arg-type]
                if parent in visited:
                    continue
                visited.add(parent)
                entity = entities.get(parent)
                if entity is None:
                    continue
                if set(entity.components) & SOLID_TYPES:
                    owners.add(parent)
                    continue
                nxt.append(parent)
        if owners:
            levels.append((depth, sorted(owners)))
        if not nxt:
            break
        frontier = nxt
    return levels


def verify(model, point_ids: Sequence[int], chain_samples: int = 3) -> dict:
    reverse = reverse_index(model.entities)
    rows: List[dict] = []
    legacy_histogram: Counter = Counter()
    full_histogram: Counter = Counter()
    differences: List[dict] = []

    for point_id in point_ids:
        levels = owners_by_level(model.entities, reverse, point_id)
        legacy_owners = list(levels[0][1]) if levels else []
        full_owners: List[int] = sorted({owner for _, owners in levels for owner in owners})
        for owner in legacy_owners:
            legacy_histogram[owner] += 1
        for owner in full_owners:
            full_histogram[owner] += 1
        row = {
            "point_id": point_id,
            "depths_with_owners": [depth for depth, _ in levels],
            "owners_by_depth": {str(depth): owners for depth, owners in levels},
            "legacy_owner_count": len(legacy_owners),
            "full_owner_count": len(full_owners),
            "legacy_owners": legacy_owners,
            "full_owners": full_owners,
        }
        rows.append(row)
        if set(legacy_owners) != set(full_owners):
            detail = classify_point(model.entities, reverse, point_id)
            chains = []
            for path in walk_up_paths(model.entities, point_id, reverse)[:chain_samples]:
                chains.append(
                    [{"eid": eid, "type": model.entities[eid].etype} for eid in path]
                )
            detail["chains"] = chains
            row["classification"] = detail
            differences.append(row)

    depth_spread = sum(1 for row in rows if len(row["depths_with_owners"]) > 1)
    return {
        "points_checked": len(rows),
        "legacy_solids": len(legacy_histogram),
        "full_solids": len(full_histogram),
        "legacy_histogram": dict(legacy_histogram.most_common()),
        "full_histogram": dict(full_histogram.most_common()),
        "points_with_multiple_owner_depths": depth_spread,
        "points_where_early_stop_would_lose_an_owner": len(differences),
        "differences": differences[:20],
        "first_owner_depth_histogram": dict(
            Counter(row["depths_with_owners"][0] for row in rows if row["depths_with_owners"])
        ),
        "per_point": rows,
    }


def main(argv: "Sequence[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Verify attribution depth structure")
    parser.add_argument("--input", required=True, help="STEP file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--compare-with",
        default="",
        help="previous step_attribute.json to compare the full histogram against",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = parse_step(args.input)
    bands, outlier_indices = point_statistics(model.points)
    point_ids = [model.point_ids[i] for i in outlier_indices]
    report = verify(model, point_ids)

    report["solid_names"] = {
        str(solid_id): entity_name(model.entities[solid_id])
        for solid_id in list(report["full_histogram"])[:20]
        if solid_id in model.entities
    }

    if args.compare_with:
        previous = json.loads(Path(args.compare_with).read_text(encoding="utf-8"))
        prior = previous.get("attribution", {}).get("solid_hit_histogram", {})
        prior_as_int = {int(k): int(v) for k, v in prior.items()}
        report["comparison_with_previous_run"] = {
            "previous_solids": len(prior_as_int),
            "current_full_solids": len(report["full_histogram"]),
            "histograms_equal": prior_as_int == report["full_histogram"],
            "only_in_previous": sorted(set(prior_as_int) - set(report["full_histogram"])),
            "only_in_current": sorted(set(report["full_histogram"]) - set(prior_as_int)),
        }

    path = out_dir / "attribution_depth_verification.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(path),
                "points_checked": report["points_checked"],
                "legacy_solids": report["legacy_solids"],
                "full_solids": report["full_solids"],
                "points_with_multiple_owner_depths": report["points_with_multiple_owner_depths"],
                "points_where_early_stop_would_lose_an_owner": report[
                    "points_where_early_stop_would_lose_an_owner"
                ],
                "first_owner_depth_histogram": report["first_owner_depth_histogram"],
                "comparison": report.get("comparison_with_previous_run"),
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
