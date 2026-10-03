import json
new = json.load(open('eth_only_v1_data_audit.json'))
old = json.load(open('v1_buggy_backup/eth_only_v1_data_audit.json'))
keys = sorted(set(new) | set(old))
for k in keys:
    a, b = old.get(k, '<absent>'), new.get(k, '<absent>')
    if a != b:
        print('CHANGED', k)
        print('   old:', a)
        print('   new:', b)
print('---- new full audit')
print(json.dumps(new, indent=1)[:2000])
