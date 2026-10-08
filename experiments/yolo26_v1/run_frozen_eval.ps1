$env:PYTHONUTF8 = "1"
Set-Location "E:\具身智能\badmin_project"
$args = @("tools\eval_yolo26_v1.py","--repo",".","--split","frozen_test","--ckpt-name","a_best","--ckpt-name","b_best","--ckpt-name","b_last","--device","cuda:0","--imgsz","1024","--conf","0.25","--ap-conf","0.001","--neg-thresholds","0.25,0.50,0.75")
& D:\_eval26\venv\Scripts\python.exe @args