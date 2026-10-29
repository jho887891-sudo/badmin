import base64,csv,hashlib,json,os,re,sys,time,math
from pathlib import Path
from urllib.error import HTTPError,URLError
from urllib.parse import urlencode
from urllib.request import Request,urlopen
from PIL import Image,ImageDraw
root=Path(sys.argv[1]); key=os.environ.get('ROBOFLOW_API_KEY')
if not key: print('API_KEY_PRESENT=no');sys.exit(3)
print('API_KEY_PRESENT=yes')
manifest=list(csv.DictReader((root/'manifest.csv').open(encoding='utf-8',newline='')))
files=[]
for row in manifest:
    src=Path(row['source_image']); rel=Path(*src.parts[-6:]) # dataset-root relative: <scene>_easy/images/train/file
    candidates=list((root/'images').rglob(src.name))
    if len(candidates)!=1: raise SystemExit(f'LOCAL_SOURCE_RESOLVE_ERROR file={src.name} count={len(candidates)}')
    image=candidates[0]
    digest=hashlib.sha256(image.read_bytes()).hexdigest()
    if digest.lower()!=row['source_sha256'].lower(): raise SystemExit(f'SOURCE_SHA_MISMATCH file={src.name}')
    row['local_image']=str(image); row['local_sha256']=digest; files.append((row,image))
print('SOURCE_HASHES_VERIFIED='+str(len(files)))
visdir=root/'visualizations';visdir.mkdir(exist_ok=True)
results=[]
for idx,(row,img) in enumerate(files,1):
    url='https://serverless.roboflow.com/badminton-serve-tracker-dataset/22?'+urlencode({'api_key':key})
    req=Request(url,data=base64.b64encode(img.read_bytes()),headers={'Content-Type':'application/x-www-form-urlencoded'},method='POST')
    try:
        with urlopen(req,timeout=90) as resp: status=resp.status; raw=resp.read()
    except HTTPError as exc: status=exc.code;raw=exc.read()
    except URLError as exc:
        rec={'index':idx,'file':img.name,'http_status':None,'error_type':type(exc.reason).__name__}
        results.append(rec);print(f'IMAGE={idx}/100 HTTP_STATUS=none ERROR_TYPE={rec["error_type"]}');break
    try: payload=json.loads(raw.decode('utf-8'))
    except Exception as exc: payload={'_parse_error':type(exc).__name__}
    if status<200 or status>=300 or '_parse_error' in payload or 'error' in payload:
        msg=str(payload.get('error') or payload.get('message') or payload.get('_parse_error') or 'request_failed').replace(key,'[REDACTED]')
        msg=re.sub(r'(?i)(api[_-]?key\s*[=:]\s*)\S+',r'\1[REDACTED]',msg)[:400]
        et=str(payload.get('type') or payload.get('error_type') or ('HTTPError' if status>=400 else 'APIError'))
        results.append({'index':idx,'file':img.name,'http_status':status,'error_type':et,'error_message':msg})
        print(f'IMAGE={idx}/100 HTTP_STATUS={status} ERROR_TYPE={et} ERROR_MESSAGE={msg}');break
    preds=[]
    for p in payload.get('predictions',[]):
        x,y,w,h=[float(p.get(k,0)) for k in ('x','y','width','height')]
        pts=p.get('points')
        preds.append({'class':p.get('class'),'confidence':float(p['confidence']) if p.get('confidence') is not None else None,'bbox_xyxy':[x-w/2,y-h/2,x+w/2,y+h/2],'mask_present':bool(pts or p.get('mask')),'mask_point_count':len(pts) if isinstance(pts,list) else None})
    im=Image.open(img).convert('RGB');draw=ImageDraw.Draw(im)
    for p in preds:
        x1,y1,x2,y2=p['bbox_xyxy'];draw.rectangle((x1,y1,x2,y2),outline=(255,40,40),width=max(2,round(max(im.size)/700)))
        conf='' if p['confidence'] is None else f" {p['confidence']:.3f}"
        draw.text((max(0,x1),max(0,y1-16)),f"{p['class']}{conf}",fill=(255,40,40),stroke_width=1,stroke_fill=(0,0,0))
    draw.rectangle((0,0,im.width,24),fill=(0,0,0));draw.text((6,5),f"{row['location']} frame={row['frame']} | {len(preds)} prediction(s)",fill=(255,255,0))
    vispath=visdir/f"infer_{idx:03d}_pred.jpg";im.save(vispath,quality=92)
    results.append({'index':idx,'location':row['location'],'time_bin_8':int(row['time_bin_8']),'frame':int(row['frame']),'source_image':row['source_image'],'image_sha256':row['local_sha256'],'http_status':status,'prediction_count':len(preds),'predictions':preds,'visualization':str(vispath)})
    print(f'IMAGE={idx}/100 HTTP_STATUS={status} PREDICTIONS={len(preds)}',flush=True)
    if idx<100: time.sleep(.08)
report={'model_id':'badminton-serve-tracker-dataset/22','endpoint':'https://serverless.roboflow.com/{dataset_id}/{version_id}','request_count':len(results),'all_http_success':len(results)==100 and all(x.get('http_status')==200 and 'prediction_count' in x for x in results),'results':results}
(root/'inference_100.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
with (root/'inference_100.jsonl').open('w',encoding='utf-8') as f:
    for x in results:f.write(json.dumps(x,ensure_ascii=False)+'\n')
with (root/'manifest_with_local_sha.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(files[0][0]));w.writeheader();w.writerows(row for row,img in files)
if not report['all_http_success']: sys.exit(5)
# comparison sheet: each tile shows source and predicted overlay side by side.
thumb_w,thumb_h=240,150; cell_w=thumb_w*2; cell_h=thumb_h+36; cols=5; rows_n=math.ceil(len(files)/cols)
sheet=Image.new('RGB',(cols*cell_w,rows_n*cell_h),(32,32,32));d=ImageDraw.Draw(sheet)
for idx,(row,img) in enumerate(files):
    x=(idx%cols)*cell_w;y=(idx//cols)*cell_h
    original=Image.open(img).convert('RGB'); overlay=Image.open(visdir/f'infer_{idx+1:03d}_pred.jpg').convert('RGB')
    original.thumbnail((thumb_w,thumb_h));overlay.thumbnail((thumb_w,thumb_h))
    sheet.paste(original,(x,y+18));sheet.paste(overlay,(x+thumb_w,y+18))
    d.text((x+3,y+2),f"{idx+1:03d} {row['location']} f{row['frame']}",fill=(255,255,255))
sheet.save(root/'comparison_sheet_100.png')
print('REQUEST_COUNT='+str(len(results)))
print('PREDICTION_TOTAL='+str(sum(x.get('prediction_count',0) for x in results)))
print('EVIDENCE='+str(root/'inference_100.json'))
print('CONTACT_SHEET='+str(root/'comparison_sheet_100.png'))
