import math
from pathlib import Path
import cv2, numpy as np
repo = Path('/home/T7/ojh/robot_sim')
src = repo / 'outputs/shuttle_capability/real_images/backgrounds'
files = sorted([p for p in src.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')])
tile = 300; cols = 6; rows = math.ceil(len(files)/cols)
sheet = np.full((rows*tile, cols*tile, 3), 20, np.uint8)
for i, f in enumerate(files):
    im = cv2.imread(str(f))
    if im is None: continue
    s = tile / max(im.shape[:2])
    im = cv2.resize(im, (int(im.shape[1]*s), int(im.shape[0]*s)))
    y, x = (i//cols)*tile, (i%cols)*tile
    sheet[y:y+im.shape[0], x:x+im.shape[1]] = im
    cv2.putText(sheet, f'#{i:02d}', (x+4,y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
out = repo / 'outputs/shuttle_capability/visualizations/backgrounds_contact_sheet.png'
cv2.imwrite(str(out), sheet)
print('backgrounds:', len(files), '->', out, sheet.shape)