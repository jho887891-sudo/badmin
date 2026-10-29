import csv, re, sys, json
from pathlib import Path
root=Path(sys.argv[1]); inv=root/'instances.csv'
rows=list(csv.DictReader(inv.open(encoding='utf-8-sig',newline='')))
by={}
for r in rows: by.setdefault(r['image'],r)
used=set()
for r in rows:
    if Path(r['image']).name in {f'{scene}_{frame:05d}.jpg' for scene in ('cab_1','cab_2','glc_1','ml_4','ticino_1') for frame in (0,1)}:
        used.add(r['image'])
# Add the two specific ml_4 files used by the 10-image probe.
used.update(r['image'] for r in rows if Path(r['image']).name in {'ml_4_00008.jpg','ticino_1_00038.jpg','ticino_1_00042.jpg'})
locations=sorted({r['location'] for r in by.values()})
quotas=[17,17,17,17,16,16]
selected=[]
for loc,quota in zip(locations,quotas):
    group=[r for r in by.values() if r['location']==loc]
    def frame(r):
        m=re.search(r'_(\d+)\.(?:jpg|jpeg|png)$',Path(r['image']).name,re.I)
        return int(m.group(1)) if m else -1
    group.sort(key=lambda r:(frame(r),r['image']))
    bins=[group[(b*len(group))//8:((b+1)*len(group))//8] for b in range(8)]
    base,extra=divmod(quota,8)
    for b,items in enumerate(bins):
        need=base+(b<extra)
        avail=[r for r in items if r['image'] not in used]
        if len(avail)<need: raise SystemExit(f'insufficient: {loc} bin {b}')
        chosen=[]
        for j in range(need):
            ix=round((j+0.5)*len(avail)/need-0.5)
            candidate=avail[ix]
            if candidate in chosen:
                candidate=next(x for x in avail if x not in chosen)
            chosen.append(candidate)
        for r in chosen:
            used.add(r['image']); selected.append({'location':loc,'time_bin_8':b,'frame':frame(r),'source_image':r['image'],'source_sha256':r['source_sha256']})
if len(selected)!=100 or len({r['source_image'] for r in selected})!=100: raise SystemExit('selection not 100 unique')
batch=root/'batch100';
with (batch/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(selected[0])); w.writeheader(); w.writerows(selected)
(base:=Path('/home/T7/dgut/robot_sim/eth_shuttle_detection'))
with (batch/'remote_paths.txt').open('w',encoding='utf-8') as f:
    for r in selected: f.write(str(Path(r['source_image']).relative_to(base)).replace('\\','/')+'\n')
print('selected',len(selected),'locations',json.dumps({loc:sum(r['location']==loc for r in selected) for loc in locations}))
print('bins',json.dumps({loc:[sum(r['location']==loc and r['time_bin_8']==b for r in selected) for b in range(8)] for loc in locations}))
