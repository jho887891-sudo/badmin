import base64, hashlib, json, os, re, sys, time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from PIL import Image, ImageDraw

root=Path(sys.argv[1]); imgs=sorted(root.glob('sample_*.jpg'))
key=os.environ.get('ROBOFLOW_API_KEY')
if not key: print('API_KEY_PRESENT=no'); sys.exit(3)
print('API_KEY_PRESENT=yes')
records=[]
for idx,img in enumerate(imgs,1):
    url='https://serverless.roboflow.com/badminton-serve-tracker-dataset/22?'+urlencode({'api_key':key})
    req=Request(url,data=base64.b64encode(img.read_bytes()),headers={'Content-Type':'application/x-www-form-urlencoded'},method='POST')
    try:
        with urlopen(req,timeout=60) as resp: status=resp.status; raw=resp.read()
    except HTTPError as exc: status=exc.code; raw=exc.read()
    except URLError as exc:
        print(f'IMAGE={img.name} HTTP_STATUS=none ERROR_TYPE={type(exc.reason).__name__}')
        records.append({'file':img.name,'http_status':None,'error_type':type(exc.reason).__name__})
        break
    try: payload=json.loads(raw.decode('utf-8'))
    except Exception as exc: payload={'_parse_error':type(exc).__name__}
    if status<200 or status>=300 or '_parse_error' in payload or 'error' in payload:
        msg=str(payload.get('error') or payload.get('message') or payload.get('_parse_error') or 'request_failed').replace(key,'[REDACTED]')
        msg=re.sub(r'(?i)(api[_-]?key\s*[=:]\s*)\S+',r'\1[REDACTED]',msg)[:400]
        et=str(payload.get('type') or payload.get('error_type') or ('HTTPError' if status>=400 else 'APIError'))
        print(f'IMAGE={img.name} HTTP_STATUS={status} ERROR_TYPE={et} ERROR_MESSAGE={msg}')
        records.append({'file':img.name,'http_status':status,'error_type':et,'error_message':msg})
        break
    preds=[]
    for p in payload.get('predictions',[]):
        x,y,w,h=[float(p.get(k,0)) for k in ('x','y','width','height')]
        pts=p.get('points')
        preds.append({'class':p.get('class'),'confidence':p.get('confidence'),'bbox_xyxy':[x-w/2,y-h/2,x+w/2,y+h/2],'mask_present':bool(pts or p.get('mask')),'mask_point_count':len(pts) if isinstance(pts,list) else None})
    im=Image.open(img).convert('RGB'); draw=ImageDraw.Draw(im)
    for p in preds:
        x1,y1,x2,y2=p['bbox_xyxy']; draw.rectangle((x1,y1,x2,y2),outline=(255,40,40),width=3)
        draw.text((max(0,x1),max(0,y1-14)),f"{p['class']} {float(p['confidence']):.3f}",fill=(255,40,40))
    draw.text((8,8),f"Roboflow model 22 | {len(preds)} prediction(s)",fill=(255,255,0),stroke_width=2,stroke_fill=(0,0,0))
    vis=root/'visualizations'; vis.mkdir(exist_ok=True); im.save(vis/(img.stem+'_pred.jpg'),quality=92)
    records.append({'file':img.name,'image_sha256':hashlib.sha256(img.read_bytes()).hexdigest(),'http_status':status,'prediction_count':len(preds),'image':payload.get('image'),'predictions':preds,'visualization':str(vis/(img.stem+'_pred.jpg'))})
    print(f'IMAGE={img.name} HTTP_STATUS={status} PREDICTIONS={len(preds)}')
    if len(records)<len(imgs): time.sleep(0.1)
report={'model_id':'badminton-serve-tracker-dataset/22','request_count':len(records),'all_http_success':len(records)==len(imgs) and all(x.get('http_status')==200 and 'predictions' in x for x in records),'results':records}
(root/'ten_image_evidence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('REQUEST_COUNT='+str(len(records)))
print('ALL_HTTP_SUCCESS='+str(report['all_http_success']))
print('EVIDENCE='+str(root/'ten_image_evidence.json'))
if not report['all_http_success']: sys.exit(5)
