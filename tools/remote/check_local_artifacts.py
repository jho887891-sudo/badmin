import csv, json, hashlib
from pathlib import Path
from collections import Counter
root = Path('.').resolve()
A = json.loads(Path('outputs/shuttle_capability/metrics/eth_only_v1_data_audit.json').read_text(encoding='utf-8'))
tr = list(csv.DictReader(open('data/eth_only_v1_train_manifest.csv', encoding='utf-8')))
va = list(csv.DictReader(open('data/eth_only_v1_val_manifest.csv', encoding='utf-8')))
print('audit', A['status'], 'problems', A['problems'])
print('audit val_locations', A['val_locations'], 'consistent', A['val_locations'] == sorted(set(r['location'] for r in va)))
print('val_location_counts', A['val_location_counts'])
print('rows train/val', len(tr), len(va), 'audit says', A['train_images'], A['val_images'])
neg = [r for r in tr if r['is_negative'] == 'True']
coco = [r for r in tr if r['location'] == 'coco_train']
print('negatives', len(neg), 'coco', len(coco), 'audit', A['train_negatives'], A['negatives_used'])
print('new columns', [c for c in tr[0] if 'equiv' in c])
print('frac', A['val_positive_fraction'], 'excluded_eval', A['excluded_eval_images'], 'sha_overlap', A['sha256_overlap'])
sd = list(csv.DictReader(open('outputs/shuttle_capability/metrics/eth_only_v1_size_distribution.csv', encoding='utf-8')))
print('size buckets', sum(int(r['boxes']) for r in sd), 'boxes /', sum(int(r['boxes']) for r in sd) == len([r for r in tr if r['is_negative'] == 'False']))
print('bucket counts', {r['bucket']: int(r['boxes']) for r in sd})
print('train sha16', hashlib.sha256(Path('data/eth_only_v1_train_manifest.csv').read_bytes()).hexdigest()[:16],
      'val sha16', hashlib.sha256(Path('data/eth_only_v1_val_manifest.csv').read_bytes()).hexdigest()[:16],
      'audit sha16', hashlib.sha256(Path('outputs/shuttle_capability/metrics/eth_only_v1_data_audit.json').read_bytes()).hexdigest()[:16])
