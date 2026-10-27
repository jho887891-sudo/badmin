"""CPU-only regression tests for direct unscaled ETH bbox Copy-Paste."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

_here = Path(__file__).resolve().parent
_repo_tool = _here.parent / 'tools' / 'generate_eth_stage1_bbox_preview.py'
sys.path.insert(0, str(_repo_tool.parent if _repo_tool.exists() else _here))
from generate_eth_stage1_bbox_preview import (
    LOCATIONS, PreviewBlocked, load_inventory, paste_bbox_crop, run, select_rows,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(root: Path, sizes: list[int]) -> Path:
    fields = ['instance_id', 'image', 'label', 'source_sha256', 'label_sha256',
              'split', 'source', 'role', 'location', 'set_dir', 'bbox_xyxy']
    records = []
    for scene, n in zip(LOCATIONS, sizes):
        for k in range(n):
            p = root / f'{scene}_{k:05d}.png'
            t = root / f'{scene}_{k:05d}.txt'
            im = Image.new('RGB', (34, 27), (9, 20, 37))
            im.putpixel((0, 0), (LOCATIONS.index(scene), k % 256, k // 256))
            for x in range(11, 16):
                for y in range(5, 13):
                    im.putpixel((x, y), (238, 230, 211))
            im.save(p)
            t.write_text('0 0.397 0.315 0.147 0.296\n', encoding='utf-8')
            records.append({
                'instance_id': f'{digest(p)}#{k}',
                'image': str(p), 'label': str(t),
                'source_sha256': digest(p), 'label_sha256': digest(t),
                'split': 'train', 'source': 'eth_main', 'role': 'eth_train',
                'location': scene, 'set_dir': f'{scene}/images/train',
                'bbox_xyxy': json.dumps([10, 4, 18, 15]),
            })
    inventory = root / 'inventory.csv'
    with inventory.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return inventory


class BboxStage1Tests(unittest.TestCase):
    def test_pixel_exact_unscaled_crop_including_background(self):
        im = Image.new('RGB', (20, 20), (2, 3, 4))
        im.putpixel((5, 6), (200, 100, 12))
        out, box = paste_bbox_crop(im, (4, 5, 8, 9), canvas_size=(24, 24),
                                   position=(9, 10), color=(20, 40, 60))
        self.assertEqual(box, (9, 10, 13, 14))
        self.assertEqual(out.crop(box).tobytes(), im.crop((4, 5, 8, 9)).tobytes())
        self.assertEqual(out.getpixel((0, 0)), (20, 40, 60))

    def test_no_truncation_allowed(self):
        im = Image.new('RGB', (20, 20))
        with self.assertRaises(PreviewBlocked):
            paste_bbox_crop(im, (4, 5, 8, 9), canvas_size=(10, 10),
                            position=(9, 9), color=(20, 40, 60))

    def test_preview_100_balances_all_scenes_and_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inv = make_fixture(root, [21, 22, 23, 24, 25, 26])
            out = root / 'preview'
            report = run(inv, out, mode='preview', count=100, seed=21,
                         bins=10, size=(96, 96), expected_count=0)
            self.assertEqual(report['generated_images'], 100)
            self.assertEqual(report['unique_instances_covered'], 100)
            self.assertEqual(report['duplicate_exposures'], 0)
            self.assertEqual(len(list((out/'images').glob('*.png'))), 100)
            self.assertEqual(len(list((out/'labels').glob('*.txt'))), 100)
            self.assertEqual(sorted(report['by_scene'].values()), [16, 16, 17, 17, 17, 17])
            with (out/'manifest.csv').open(encoding='utf-8') as f:
                records = list(csv.DictReader(f))
            self.assertTrue(all(r['scale'] == '1.0' for r in records))
            self.assertTrue(all(r['foreground_method'] == 'bbox_crop_no_mask' for r in records))
            self.assertEqual(len(set(r['instance_id'] for r in records)), 100)
            self.assertTrue((out/'contact_sheet.png').is_file())
            for row in records[:6]:
                with Image.open(row['source_image']) as src, Image.open(out/row['image_path']) as result:
                    orig = src.convert('RGB').crop(json.loads(row['source_bbox_xyxy']))
                    patch = result.convert('RGB').crop(json.loads(row['output_bbox_xyxy']))
                    self.assertEqual(orig.tobytes(), patch.tobytes())

    def test_reproducible_png_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inv = make_fixture(root, [3] * 6)
            a = run(inv, root/'a', count=12, seed=42, bins=3,
                    size=(48, 48), expected_count=0)
            b = run(inv, root/'b', count=12, seed=42, bins=3,
                    size=(48, 48), expected_count=0)
            self.assertEqual(a['manifest_sha256'], b['manifest_sha256'])
            self.assertEqual(a['contact_sheet_sha256'], b['contact_sheet_sha256'])
            self.assertEqual([(root/'a'/'images'/f'{i:06d}.png').read_bytes()
                              for i in range(1,13)],
                             [(root/'b'/'images'/f'{i:06d}.png').read_bytes()
                              for i in range(1,13)])

    def test_full_contains_every_instance_exactly_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inv = make_fixture(root, [3, 4, 3, 4, 3, 4])
            audit = run(inv, root/'full', mode='full', seed=7, bins=2,
                        size=(45, 45), expected_count=0)
            self.assertTrue(audit['full_unique_coverage'])
            self.assertEqual(audit['duplicate_exposures'], 0)
            self.assertEqual(audit['generated_images'], 21)

    def test_balanced_full_repeats_instead_of_dropping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inv = make_fixture(root, [4, 6, 4, 6, 4, 6])
            audit = run(inv, root/'balance', mode='balanced-full', seed=7, bins=2,
                        size=(45, 45), expected_count=0)
            self.assertTrue(audit['full_unique_coverage'])
            self.assertGreater(audit['duplicate_exposures'], 0)
            self.assertEqual(len(set(audit['by_scene'].values())), 1)
            self.assertEqual(len(set(audit['by_scene_time_bin'].values())), 1)

    def test_fail_on_nontrain_source_or_bad_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inv = make_fixture(root, [1] * 6)
            with inv.open(encoding='utf-8') as fh:
                rows = list(csv.DictReader(fh))
            rows[0]['source'] = 'eval'
            with inv.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader(); writer.writerows(rows)
            with self.assertRaises(PreviewBlocked):
                load_inventory(inv, expected_count=0)
            rows[0]['source'] = 'eth_main'
            rows[0]['source_sha256'] = '0'*64
            with inv.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader(); writer.writerows(rows)
            with self.assertRaises(PreviewBlocked):
                run(inv, root/'out', count=6, size=(45, 45), expected_count=0)
            self.assertFalse((root/'out').exists())

    def test_existing_destination_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inv = make_fixture(root, [1] * 6)
            (root/'out').mkdir()
            with self.assertRaises(PreviewBlocked):
                run(inv, root/'out', count=6, size=(45, 45), expected_count=0)


if __name__ == '__main__':
    unittest.main()
