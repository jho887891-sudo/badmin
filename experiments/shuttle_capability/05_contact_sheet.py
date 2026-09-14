import argparse, csv, math
from pathlib import Path
import cv2, numpy as np
p = argparse.ArgumentParser(); p.add_argument('--repo', default='/home/T7/ojh/robot_sim'); p.add_argument('--cols', type=int, default=8); p.add_argument('--tile', type=int, default=260)
a = p.parse_args()
repo = Path(a.repo)
src = repo / 'outputs/shuttle_capability/real_images/raw'
files = sorted(src.glob('*.jpg')) + sorted(src.glob('*.png'))
cols = a.cols; tile = a.tile
rows = math.ceil(len(files) / cols)
sheet = np.full((rows * tile, cols * tile, 3), 30, np.uint8)
for i, f in enumerate(files):
    img = cv2.imread(str(f))
    if img is None: continue
    h, w = img.shape[:2]; s = tile / max(h, w)
    r = cv2.resize(img, (int(w*s), int(h*s)))
    y, x = (i // cols) * tile, (i % cols) * tile
    sheet[y:y+r.shape[0], x:x+r.shape[1]] = r
    cv2.putText(sheet, f'#{i:02d}', (x+4, y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
out = repo / 'outputs/shuttle_capability/visualizations/real_images_contact_sheet.png'
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), sheet)
print('files:', len(files), '-> ', out, sheet.shape)