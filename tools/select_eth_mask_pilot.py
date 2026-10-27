#!/usr/bin/env python3
"""Deterministic size-stratified ETH Mask QA pilot, with R0 integrity gate.

Read-only source: no raw image/label reads, training, or Mask approvals.
Produces a small inventory compatible with run_eth_mask_candidates.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


class PilotBlocked(RuntimeError):
    """A source/selection/overwrite gate failed."""


FIELDS_REQUIRED = frozenset({
    'instance_id', 'image', 'label', 'source_sha256', 'label_sha256',
    'split', 'source', 'role', 'location', 'bbox_xyxy', 'bbox_short', 'bbox_long',
})
BANDS = ('lt4', '4to9', '10to23', 'ge24')
SHA_INDEX = re.compile(r'^[0-9a-f]{64}#\d+$')
SHA = re.compile(r'^[0-9a-f]{64}$')


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for buf in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def _band(value: int) -> str:
    if value < 4:
        return 'lt4'
    if value < 10:
        return '4to9'
    if value < 24:
        return '10to23'
    return 'ge24'


def select_mask_pilot(inventory_csv: str | Path, r0_audit_json: str | Path,
                      output_dir: str | Path, *, pilot_count: int = 120, seed: int = 42) -> dict:
    src, audit_path, out = Path(inventory_csv), Path(r0_audit_json), Path(output_dir)
    if out.exists():
        raise PilotBlocked('output_directory_already_exists: ' + str(out))
    if pilot_count < 4 or pilot_count > 1024:
        raise PilotBlocked('pilot_count_out_of_range: 4..1024')
    if not src.is_file() or not audit_path.is_file():
        raise PilotBlocked('R0_inventory_or_audit_missing')
    try:
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise PilotBlocked('R0_audit_unreadable') from exc
    if (audit.get('source_gate') != 'PASS' or
            audit.get('status') != 'CANDIDATES_PENDING_REVIEW' or
            audit.get('next_gate') != 'HUMAN_MASK_REVIEW_REQUIRED'):
        raise PilotBlocked('R0_gate_not_pass')
    expected = str(audit.get('inventory_csv_sha256') or '').lower()
    actual = _sha(src)
    if not SHA.fullmatch(expected) or actual != expected:
        raise PilotBlocked('R0_inventory_sha256_mismatch')

    with src.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        if not FIELDS_REQUIRED.issubset(headers):
            raise PilotBlocked('R0_inventory_schema_missing')
        rows = list(reader)
    if len(rows) < pilot_count:
        raise PilotBlocked('R0_inventory_too_small')
    if audit.get('counts', {}).get('usable_instances') is not None and (
            int(audit['counts']['usable_instances']) != len(rows)):
        raise PilotBlocked('R0_inventory_count_mismatch')
    seen = set()
    pools = {band: [] for band in BANDS}
    for row in rows:
        iid = row.get('instance_id', '').strip().lower()
        if not SHA_INDEX.fullmatch(iid) or iid in seen:
            raise PilotBlocked('duplicate_or_invalid_instance_id')
        seen.add(iid)
        if (row.get('split') != 'train' or row.get('source') != 'eth_main' or
                row.get('role') != 'eth_train' or row.get('status') != 'PENDING_REVIEW'):
            raise PilotBlocked('unauthorized_source_role_split_or_review_state')
        if (not SHA.fullmatch(str(row.get('source_sha256') or '').lower()) or
                not SHA.fullmatch(str(row.get('label_sha256') or '').lower())):
            raise PilotBlocked('source_or_label_sha256_missing')
        try:
            short, long_side = int(row['bbox_short']), int(row['bbox_long'])
        except (ValueError, TypeError) as exc:
            raise PilotBlocked('invalid_bbox_size') from exc
        if short < 1 or long_side < short:
            raise PilotBlocked('invalid_bbox_size')
        try:
            bbox = json.loads(row['bbox_xyxy'])
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError('bbox')
        except ValueError as exc:
            raise PilotBlocked('invalid_bbox_xyxy') from exc
        pools[_band(short)].append(row)

    # Stable across CSV orderings and platforms; no code can silently choose
    # easiest-to-segment examples simply because they appear first in the file.
    for band in BANDS:
        pools[band].sort(key=lambda r: hashlib.sha256(
            f"{seed}|{r['instance_id']}".encode('utf-8')).hexdigest())
    chosen = []
    offsets = {b: 0 for b in BANDS}
    while len(chosen) < pilot_count:
        progressed = False
        for band in BANDS:
            if len(chosen) >= pilot_count:
                break
            if offsets[band] >= len(pools[band]):
                continue
            chosen.append(pools[band][offsets[band]])
            offsets[band] += 1
            progressed = True
        if not progressed:
            raise PilotBlocked('not_enough_eligible_rows')

    out.mkdir(parents=True)
    pilot_path = out / 'pilot_instances.csv'
    with pilot_path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(chosen)
    counts = Counter(_band(int(r['bbox_short'])) for r in chosen)
    locations = Counter(str(r.get('location') or 'UNKNOWN') for r in chosen)
    report = {
        'status': 'PILOT_READY_FOR_MASK_PROPOSALS',
        'source_inventory': str(src),
        'source_inventory_sha256': actual,
        'source_r0_audit': str(audit_path),
        'source_r0_audit_sha256': _sha(audit_path),
        'source_rows': len(rows),
        'selected_rows': len(chosen),
        'selection_seed': seed,
        'selection_rule': 'size-stratified deterministic round-robin, all four short-side bands',
        'bands_short_source_px': {'lt4': '<4', '4to9': '4..9', '10to23': '10..23', 'ge24': '>=24'},
        'selected_by_band': {b: counts[b] for b in BANDS},
        'available_by_band': {b: len(pools[b]) for b in BANDS},
        'selected_by_location': dict(sorted(locations.items())),
        'pilot_inventory': str(pilot_path),
        'pilot_sha256': _sha(pilot_path),
        'human_approved_masks': 0,
        'training_permitted': False,
        'optimizer_steps': 0,
        'next_gate': 'MASK_CANDIDATES_AND_HUMAN_REVIEW',
    }
    (out / 'pilot_audit.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--r0-audit', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pilot-count', type=int, default=120)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    try:
        report = select_mask_pilot(args.inventory, args.r0_audit,
                                   args.output_dir, pilot_count=args.pilot_count, seed=args.seed)
    except (PilotBlocked, OSError) as exc:
        print(json.dumps({'status': 'BLOCKED_R1_PILOT', 'reason': str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
