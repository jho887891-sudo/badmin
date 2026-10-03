import csv, json
A = json.load(open('eth_only_v1_data_audit.json'))
print('status', A['status'], 'train', A['train_images'], 'val', A['val_images'], 'neg', A['train_negatives'])
print('val_locations', A['val_locations'])
print('val_counts', A['val_location_counts'])
print('frac', A['val_positive_fraction'], 'excluded_eval', A['excluded_eval_images'])
print('val_locs_consistent', A['val_locations'] == sorted(A['val_location_counts']))
print('--- size_distribution.csv')
print(open('eth_only_v1_size_distribution.csv').read())
rows = list(csv.DictReader(open('eth_only_v1_train_manifest.csv')))
print('train_rows', len(rows), 'cols', [c for c in rows[0] if 'equiv' in c or c == 'n_boxes'])
print('sample', {k: rows[0][k] for k in ('location', 'n_boxes', 'width', 'height', 'equiv_size_640',
                                         'median_equiv_size_640', 'equiv_sizes')})
nb = [int(r['n_boxes']) for r in rows if r['is_negative'] == 'False']
print('pos_imgs', len(nb), 'pos_boxes', sum(nb))
print('negatives_with_boxes', sum(1 for r in rows if r['is_negative'] == 'True' and int(r['n_boxes']) != 0))
vrows = list(csv.DictReader(open('eth_only_v1_val_manifest.csv')))
print('val_rows', len(vrows), 'val_locs', sorted(set(r['location'] for r in vrows)))
print('val_loc_counts', {l: sum(1 for r in vrows if r['location'] == l) for l in sorted(set(r['location'] for r in vrows))})
