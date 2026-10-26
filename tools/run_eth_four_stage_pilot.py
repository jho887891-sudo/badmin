#!/usr/bin/env python3
"""CPU-only deterministic fixture covering Stage 1, 2 and 4.

This exercises the data-conversion contracts but is NOT real ETH extraction,
training, or independent GPU/real-world model evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from eth_four_stage_solid import create_stage1_sample
from eth_four_stage_degrade import degrade_stage2_sample
from eth_four_stage_environments import make_stage4_sample


def generate_fixture_pilot(seed:int=42) -> dict:
    rgb=np.full((20,20,3),242,dtype=np.uint8)
    alpha=np.zeros((20,20),dtype=np.float32)
    alpha[3:17,7:13]=1
    s1=create_stage1_sample(rgb,alpha,canvas_size=(96,64),color=(25,42,76),center=(48,32),seed=seed)
    s2=degrade_stage2_sample(s1,{'motion_blur_px':3,'motion_angle_deg':20,
                                 'gamma':1.1,'noise_sigma':0.2},seed=seed)
    bg=np.full((64,96,3),75,dtype=np.uint8)
    bg[:12,:,:]=[170,170,192]
    fg={'sample_id':'cpu-only-eth-fixture','rgb':rgb,'alpha':alpha,
        'source':'eth_train','source_sha256':'F'*64,'qa_status':'APPROVED',
        'reviewed':True,'reviewer_id':'CPU_FIXTURE_NOT_REAL','review_method':'synthetic_fixture'}
    env={'background_id':'cpu-only-venue-fixture','image':bg,'source':'real_venue_train',
         'source_sha256':'B'*64,'qa_status':'APPROVED','split':'train',
         'reviewer_id':'fixture-automated-not-real-review','no_shuttle_reviewed':True,
         'environment_type':'fixture_lamp','clean_roi':[5,15,90,60]}
    s4=make_stage4_sample(fg,env,{'center_x':45,'center_y':35,'scale':1.0},seed=seed)
    return {'stage1':s1,'stage2':s2,'stage4':s4}


def _sha(x:bytes)->str:
    return hashlib.sha256(x).hexdigest()


def save_pilot(samples:dict,output_dir:str|Path)->dict:
    out=Path(output_dir)
    if out.exists():raise FileExistsError('refuse to overwrite pilot output: '+str(out))
    out.mkdir(parents=True)
    records=[]
    for stage,sample in sorted(samples.items()):
        imgpath=out/f'{stage}.png'
        labelpath=out/f'{stage}.txt'
        Image.fromarray(sample['image']).save(imgpath,format='PNG')
        labelpath.write_text(sample['label'],encoding='utf-8')
        records.append({'stage':stage,'image_path':imgpath.name,'label_path':labelpath.name,
                        'image_sha256':_sha(imgpath.read_bytes()),
                        'label_sha256':_sha(labelpath.read_bytes()),
                        'bbox_xyxy':sample['metadata']['bbox_xyxy'],
                        'seed':sample['metadata']['seed']})
    report={'status':'CPU_FIXTURE_ONLY','real_data_used':False,
            'optimizer_steps':0,'real_recall':'UNKNOWN','records':records}
    (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture-output',required=True)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args(argv)
    print(json.dumps(save_pilot(generate_fixture_pilot(args.seed),args.fixture_output),ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
