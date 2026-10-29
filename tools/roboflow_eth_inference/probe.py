import base64, hashlib, json, os, re, sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

image = Path(sys.argv[1])
out = Path(sys.argv[2])
key = os.environ.get('ROBOFLOW_API_KEY')
if not key:
    print('API_KEY_PRESENT=no')
    sys.exit(3)
print('API_KEY_PRESENT=yes')
body = base64.b64encode(image.read_bytes())
url = 'https://serverless.roboflow.com/badminton-serve-tracker-dataset/22?' + urlencode({'api_key': key})
req = Request(url, data=body, headers={'Content-Type': 'application/x-www-form-urlencoded'}, method='POST')
status = None
try:
    with urlopen(req, timeout=60) as response:
        status = response.status
        raw = response.read()
except HTTPError as exc:
    status = exc.code
    raw = exc.read()
except URLError as exc:
    print('HTTP_STATUS=none')
    print('ERROR_TYPE=' + type(exc.reason).__name__)
    print('ERROR_MESSAGE=' + re.sub(r'(?i)(api[_-]?key\s*[=:]\s*)\S+', r'\1[REDACTED]', str(exc.reason))[:300])
    evidence = {'request_count': 1, 'http_status': None, 'error_type': type(exc.reason).__name__}
    (out / 'probe_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    sys.exit(4)
try:
    payload = json.loads(raw.decode('utf-8'))
except Exception as exc:
    payload = {'_parse_error': type(exc).__name__}
print('HTTP_STATUS=' + str(status))
if status < 200 or status >= 300 or '_parse_error' in payload or 'error' in payload:
    msg = str(payload.get('error') or payload.get('message') or payload.get('_parse_error') or 'request_failed')
    msg = msg.replace(key, '[REDACTED]')
    msg = re.sub(r'(?i)(api[_-]?key\s*[=:]\s*)\S+', r'\1[REDACTED]', msg)[:400]
    etype = str(payload.get('type') or payload.get('error_type') or ('HTTPError' if status >= 400 else payload.get('_parse_error','APIError')))
    print('ERROR_TYPE=' + etype)
    print('ERROR_MESSAGE=' + msg)
    evidence = {'request_count': 1, 'http_status': status, 'error_type': etype, 'error_message': msg}
    (out / 'probe_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    sys.exit(5)
rows=[]
for p in payload.get('predictions', []):
    x,y,w,h = [float(p.get(k, 0)) for k in ('x','y','width','height')]
    rows.append({'class': p.get('class'), 'confidence': p.get('confidence'), 'bbox_xyxy':[x-w/2,y-h/2,x+w/2,y+h/2], 'mask_present': bool(p.get('points') or p.get('mask')), 'mask_point_count': len(p.get('points', [])) if isinstance(p.get('points'), list) else None})
evidence={'request_count':1,'http_status':status,'error_type':None,'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'prediction_count':len(rows),'image':payload.get('image'),'predictions':rows}
(out / 'probe_evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
print('PREDICTION_COUNT=' + str(len(rows)))
print('CLASSES=' + json.dumps(sorted({str(p['class']) for p in rows}),ensure_ascii=False))
print('MASK_PREDICTIONS=' + str(sum(p['mask_present'] for p in rows)))
print('EVIDENCE=' + str(out / 'probe_evidence.json'))
