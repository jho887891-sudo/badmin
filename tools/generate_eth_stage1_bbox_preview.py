#!/usr/bin/env python3
"""ETH Stage 1: pixel-exact bbox cropping onto non-white solid backgrounds.

The verified ETH TRAIN inventory is the only source. No GrabCut, masks, resize,
alpha matting, training or modification of original images.
Frame-index bins are only a capture-progress proxy, not physical timestamps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

LOCATIONS = ('cab_1', 'cab_2', 'glc_1', 'ml_4', 'ticino_1', 'ticino_2')
PALETTE = (
    (22, 39, 65), (40, 69, 60), (73, 37, 59), (52, 59, 86),
    (93, 62, 35), (32, 81, 88), (96, 58, 76), (69, 75, 46),
)
FIELDS = (
    'ordinal', 'instance_id', 'location', 'set_dir', 'frame_index', 'time_bin',
    'time_basis', 'source_image', 'source_sha256', 'source_label',
    'label_sha256', 'source_bbox_xyxy', 'crop_width', 'crop_height',
    'paste_x', 'paste_y', 'output_bbox_xyxy', 'foreground_method',
    'scale', 'background_rgb', 'image_path', 'label_path',
    'image_sha256', 'output_label_sha256', 'repeated_source_instance',
)


class PreviewBlocked(ValueError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _path(value: str, root: Path | None) -> Path:
    p = Path(value)
    return (root / p) if root is not None and not p.is_absolute() else p


def _frame_index(imgpath: str) -> int:
    groups = re.findall(r'\d+', Path(imgpath).stem)
    if not groups:
        raise PreviewBlocked('no numeric frame index: ' + imgpath)
    return int(groups[-1])


def _bbox(raw: str) -> tuple[int, int, int, int]:
    try:
        seq = json.loads(raw)
        if not isinstance(seq, list) or len(seq) != 4 or any(type(x) is not int for x in seq):
            raise ValueError
        x0, y0, x1, y1 = seq
        if min(x0, y0) < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError
        return x0, y0, x1, y1
    except (ValueError, TypeError) as exc:
        raise PreviewBlocked('invalid inventory bbox_xyxy: ' + raw) from exc


def load_inventory(inventory: Path, *, expected_count: int = 13992, bins: int = 10) -> list[dict]:
    if bins < 1:
        raise PreviewBlocked('time-bins must be >= 1')
    if not inventory.is_file():
        raise PreviewBlocked('inventory not found: ' + str(inventory))
    with inventory.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'instance_id', 'image', 'label', 'source_sha256', 'label_sha256',
                    'split', 'source', 'role', 'location', 'set_dir', 'bbox_xyxy'}
        if not required.issubset(reader.fieldnames or []):
            raise PreviewBlocked('missing inventory CSV columns: ' +
                                 ','.join(sorted(required - set(reader.fieldnames or []))))
        rows = [dict(row) for row in reader]
    if not rows or (expected_count and len(rows) != expected_count):
        raise PreviewBlocked(f'inventory count {len(rows)}, expected {expected_count or "nonzero"}')
    seen = set()
    bounds = defaultdict(list)
    for row in rows:
        iid = row['instance_id']
        if not iid or iid in seen:
            raise PreviewBlocked('duplicate or empty instance_id: ' + iid)
        seen.add(iid)
        if (row['split'], row['source'], row['role']) != ('train', 'eth_main', 'eth_train'):
            raise PreviewBlocked('non-ETH-TRAIN source: ' + iid)
        if row['location'] not in LOCATIONS:
            raise PreviewBlocked('unexpected ETH location: ' + row['location'])
        if not row['set_dir']:
            raise PreviewBlocked('missing set_dir for ' + iid)
        row['_bbox'] = _bbox(row['bbox_xyxy'])
        row['_frame_index'] = _frame_index(row['image'])
        bounds[(row['location'], row['set_dir'])].append(row['_frame_index'])
    for row in rows:
        frame_list = bounds[(row['location'], row['set_dir'])]
        minimum, maximum = min(frame_list), max(frame_list)
        row['_time_bin'] = min(bins - 1, (row['_frame_index'] - minimum) * bins //
                               (maximum - minimum + 1))
    return rows


def select_rows(rows: list[dict], *, mode: str, count: int, seed: int,
                bins: int, max_outputs: int = 100000) -> list[dict]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['location'], row['_time_bin'])].append(row)
    for bucket in groups.values():
        bucket.sort(key=lambda r: r['instance_id'])
        rng.shuffle(bucket)
    if mode == 'full':
        result = list(rows)
        rng.shuffle(result)
        return result
    if mode == 'balanced-full':
        missing = [(s, t) for s in LOCATIONS for t in range(bins) if not groups[(s, t)]]
        if missing:
            raise PreviewBlocked('balanced-full requires every scene/time cell populated; '
                                 f'empty cells: {missing}')
        target = max(map(len, groups.values()))
        total = len(LOCATIONS) * bins * target
        if total > max_outputs:
            raise PreviewBlocked(f'balanced-full would emit {total} files; '
                                 f'max-outputs={max_outputs}; explicit budget required')
        result = []
        for scene in LOCATIONS:
            for bin_id in range(bins):
                bucket = groups[(scene, bin_id)]
                result.extend(bucket)
                result.extend(rng.choice(bucket) for _ in range(target - len(bucket)))
        rng.shuffle(result)
        return result
    if mode != 'preview':
        raise PreviewBlocked('unknown mode ' + mode)
    if count < 1 or count > len(rows):
        raise PreviewBlocked('preview count must be in [1, unique_instance_count]')
    scenes = list(LOCATIONS)
    rng.shuffle(scenes)
    per_scene = {scene: count // len(scenes) for scene in scenes}
    for scene in scenes[:count % len(scenes)]:
        per_scene[scene] += 1
    result = []
    for scene in scenes:
        quota = per_scene[scene]
        active = {t: list(groups[(scene, t)]) for t in range(bins) if groups[(scene, t)]}
        if sum(map(len, active.values())) < quota:
            raise PreviewBlocked(f'insufficient unique inventory instances for {scene}: {quota}')
        picked = Counter()
        for _ in range(quota):
            available = [t for t, bucket in active.items() if bucket]
            minimum = min(picked[t] for t in available)
            choice = rng.choice([t for t in available if picked[t] == minimum])
            result.append(active[choice].pop())
            picked[choice] += 1
    rng.shuffle(result)
    return result


def paste_bbox_crop(image: Image.Image, bbox: tuple[int, int, int, int], *,
                    canvas_size: tuple[int, int], position: tuple[int, int],
                    color: tuple[int, int, int]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Source crop and pasted output retain identical RGB bytes."""
    im = image.convert('RGB')
    x0, y0, x1, y1 = bbox
    if x1 > im.width or y1 > im.height or min(x0, y0) < 0 or x1 <= x0 or y1 <= y0:
        raise PreviewBlocked('source bbox out of actual image bounds')
    crop = im.crop(bbox)
    W, H = canvas_size
    left, top = position
    if W <= 0 or H <= 0 or min(left, top) < 0 or left + crop.width > W or top + crop.height > H:
        raise PreviewBlocked('source crop cannot fit fully inside canvas')
    if len(color) != 3 or min(color) < 0 or max(color) > 255 or all(v >= 230 for v in color):
        raise PreviewBlocked('background must be valid non-near-white RGB')
    canvas = Image.new('RGB', canvas_size, color)
    canvas.paste(crop, (left, top))
    if canvas.crop((left, top, left + crop.width, top + crop.height)).tobytes() != crop.tobytes():
        raise PreviewBlocked('pixel-exact copy check failed')
    return canvas, (left, top, left + crop.width, top + crop.height)


def _label(box: tuple[int, int, int, int], size: tuple[int, int]) -> str:
    x0, y0, x1, y1 = box
    W, H = size
    return f'0 {(x0+x1)/(2*W):.9f} {(y0+y1)/(2*H):.9f} {(x1-x0)/W:.9f} {(y1-y0)/H:.9f}\n'


def _contact_sheet(paths: list[Path], records: list[dict], out: Path) -> None:
    """Overview and nearest-neighbor zoom ONLY in review sheet, never in saved images."""
    tile_w, tile_h, cols = 268, 238, 10
    canvas = Image.new('RGB', (cols * tile_w, ((len(paths)+cols-1)//cols)*tile_h), (235, 235, 235))
    d = ImageDraw.Draw(canvas)
    for k, (path, row) in enumerate(zip(paths, records)):
        xx, yy = (k % cols) * tile_w, (k // cols) * tile_h
        with Image.open(path) as img:
            full = img.copy()
        thumb = full.copy()
        thumb.thumbnail((122, 156), Image.Resampling.NEAREST)
        canvas.paste(thumb, (xx + (125-thumb.width)//2, yy+33))
        x0, y0, x1, y1 = json.loads(row['output_bbox_xyxy'])
        pad = max(3, max(x1-x0, y1-y0)//2)
        roi = full.crop((max(0,x0-pad),max(0,y0-pad),min(full.width,x1+pad),min(full.height,y1+pad)))
        if roi.width and roi.height:
            enlarged = roi.resize((122, 156), Image.Resampling.NEAREST)
            canvas.paste(enlarged, (xx+137, yy+33))
        d.text((xx+4, yy+5), f"#{k+1:03d} {row['location']} T{row['time_bin']} BBOX", fill=(10, 10, 10))
        d.text((xx+7, yy+201), 'Original-size canvas | zoom review', fill=(45, 45, 45))
    canvas.save(out, format='PNG')


def run(inventory: Path, output_dir: Path, *, mode: str = 'preview', count: int = 100,
        seed: int = 42, bins: int = 10, size: tuple[int, int] = (1024, 1024),
        image_root: Path | None = None, expected_count: int = 13992,
        max_outputs: int = 100000) -> dict[str, Any]:
    if output_dir.exists():
        raise PreviewBlocked('refusing to overwrite existing output: ' + str(output_dir))
    if min(size) < 1:
        raise PreviewBlocked('invalid canvas dimensions')
    rows = load_inventory(inventory, expected_count=expected_count, bins=bins)
    selected = select_rows(rows, mode=mode, count=count, seed=seed, bins=bins,
                           max_outputs=max_outputs)
    checked = set()
    for row in selected:
        if row['instance_id'] in checked:
            continue
        checked.add(row['instance_id'])
        for field, digest in [('image', 'source_sha256'), ('label', 'label_sha256')]:
            path = _path(row[field], image_root)
            if not path.is_file() or sha256(path).lower() != row[digest].lower():
                raise PreviewBlocked(f'{field} missing or SHA256 mismatch for {row["instance_id"]}: {path}')
        with Image.open(_path(row['image'], image_root)) as im:
            x0, y0, x1, y1 = row['_bbox']
            if x1 > im.width or y1 > im.height or x1-x0 > size[0] or y1-y0 > size[1]:
                raise PreviewBlocked('invalid or oversized crop: ' + row['instance_id'])
    rng = random.Random(seed + 371)
    (output_dir / 'images').mkdir(parents=True)
    (output_dir / 'labels').mkdir()
    metadata = []
    paths = []
    usage = Counter()
    for k, row in enumerate(selected, 1):
        with Image.open(_path(row['image'], image_root)) as im:
            rgb = im.convert('RGB')
        x0, y0, x1, y1 = row['_bbox']
        cw, ch = x1-x0, y1-y0
        left, top = rng.randrange(size[0]-cw+1), rng.randrange(size[1]-ch+1)
        color = PALETTE[rng.randrange(len(PALETTE))]
        canvas, outbox = paste_bbox_crop(rgb, row['_bbox'], canvas_size=size,
                                         position=(left, top), color=color)
        image_rel = f'images/{k:06d}.png'
        label_rel = f'labels/{k:06d}.txt'
        image_path, label_path = output_dir / image_rel, output_dir / label_rel
        canvas.save(image_path, format='PNG')
        label_path.write_text(_label(outbox, size), encoding='utf-8')
        usage[row['instance_id']] += 1
        record = {
            'ordinal': k, 'instance_id': row['instance_id'],
            'location': row['location'], 'set_dir': row['set_dir'],
            'frame_index': row['_frame_index'], 'time_bin': row['_time_bin'],
            'time_basis': 'frame_index_equal_range_NOT_verified_clock_time',
            'source_image': row['image'], 'source_sha256': row['source_sha256'],
            'source_label': row['label'], 'label_sha256': row['label_sha256'],
            'source_bbox_xyxy': json.dumps(row['_bbox']),
            'crop_width': cw, 'crop_height': ch,
            'paste_x': left, 'paste_y': top,
            'output_bbox_xyxy': json.dumps(outbox),
            'foreground_method': 'bbox_crop_no_mask', 'scale': '1.0',
            'background_rgb': json.dumps(color),
            'image_path': image_rel, 'label_path': label_rel,
            'image_sha256': sha256(image_path),
            'output_label_sha256': sha256(label_path),
            'repeated_source_instance': usage[row['instance_id']] > 1,
        }
        metadata.append(record)
        paths.append(image_path)
    with (output_dir/'manifest.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(metadata)
    _contact_sheet(paths[:100], metadata[:100], output_dir/'contact_sheet.png')
    per_scene = Counter(r['location'] for r in metadata)
    per_cell = Counter(f"{r['location']}|T{r['time_bin']}" for r in metadata)
    expected_all = {r['instance_id'] for r in rows}
    covered = set(usage)
    report = {
        'status': 'PASS_DATA_GENERATION_NOT_TRAINING', 'mode': mode,
        'inventory_path': str(inventory), 'inventory_sha256': sha256(inventory),
        'inventory_instances': len(expected_all), 'generated_images': len(metadata),
        'unique_instances_covered': len(covered),
        'full_unique_coverage': covered == expected_all,
        'duplicate_exposures': len(metadata)-len(covered),
        'foreground_method': 'bbox_crop_no_mask_no_resize',
        'mask_used': False, 'model_trained': False, 'optimizer_steps': 0,
        'image_size': list(size), 'scale': 1.0, 'seed': seed,
        'time_binning_method': 'frame_index_equal_range_NOT_verified_clock_time',
        'time_bins_per_sequence': bins,
        'by_scene': dict(sorted(per_scene.items())),
        'by_scene_time_bin': dict(sorted(per_cell.items())),
        'balance_limitation': ('complete source coverage does NOT imply equal scene/time exposure'
                               if mode == 'full' else
                               'only nonempty scene/time cells are sampled' if mode == 'preview' else
                               'balanced within observed and fully populated scene/time cells'),
        'visual_quality_review': 'UNKNOWN_PENDING_HUMAN_REVIEW',
        'manifest_sha256': sha256(output_dir/'manifest.csv'),
        'contact_sheet_sha256': sha256(output_dir/'contact_sheet.png'),
        'contact_sheet_scope': f'first {min(100, len(metadata))} generated images',
    }
    with (output_dir/'audit.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    with (output_dir/'SHA256SUMS').open('x', encoding='utf-8') as f:
        for file in sorted(p for p in output_dir.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'):
            f.write(f'{sha256(file)}  {file.relative_to(output_dir).as_posix()}\n')
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--image-root', type=Path, default=None)
    p.add_argument('--mode', choices=('preview', 'full', 'balanced-full'), default='preview')
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--time-bins', type=int, default=10)
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--height', type=int, default=1024)
    p.add_argument('--expected-instances', type=int, default=13992,
                   help='use 0 only for fixture tests')
    p.add_argument('--max-outputs', type=int, default=100000)
    args = p.parse_args(argv)
    try:
        report = run(args.inventory, args.output_dir, mode=args.mode, count=args.count,
                     seed=args.seed, bins=args.time_bins, size=(args.width, args.height),
                     image_root=args.image_root, expected_count=args.expected_instances,
                     max_outputs=args.max_outputs)
    except (PreviewBlocked, OSError, ValueError) as exc:
        print(json.dumps({'status': 'BLOCKED', 'reason': str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
