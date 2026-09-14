"""Fixed CPU-only input intervention; development diagnosis, not model selection AP."""
import json
import os
import sys
import time
import argparse
from pathlib import Path

os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'yolo11_dsdam_deploy'))
import cv2
import numpy as np
import torch
torch.set_num_threads(1)
cv2.setNumThreads(1)
from ultralytics import YOLO
from analyze_detection_errors import BASE,ROOT,OUT,ious,match


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--matched-val-preprocess',action='store_true')
    args=parser.parse_args()
    dest=OUT/('cpu_input_probe_v2' if args.matched_val_preprocess else 'cpu_input_probe_v1')
    dest.mkdir(exist_ok=False)
    objects=json.loads((OUT/'failure_mechanisms_v1/objects.json').read_text())
    selected=[];used=set()
    for group in ('ir_only','C_only','both'):
        for prefix in ('01','15'):
            candidates=sorted((r for r in objects if r['group']==group and r['prefix']==prefix),
                              key=lambda r:(r['image'],r['gt_index']))
            count=0
            for r in candidates:
                if r['image'] in used:continue
                selected.append(r);used.add(r['image']);count+=1
                if count==4:break
            assert count==4,(group,prefix,count)
    (dest/'protocol.json').write_text(json.dumps({'targets':selected,
        'variants':['original','blend_0.25IR','moment_control'],
        'control':'C globally adjusted to blended image mean/std, then clipped; no labels used in transforms.',
        'scope':'24 outcome-stratified images, exploratory selected sample; no full-set AP claim.',
        'device':'cpu','threads':1,'confidence':0.001,'nms_iou':0.7,'match_confidence':0.25,
        'match_iou':0.5,'rect':not args.matched_val_preprocess,'batch':1,
        'matched_val_preprocess':args.matched_val_preprocess,
        'expected_canvas':'640x640' if args.matched_val_preprocess else '512x640'},indent=2))
    model=YOLO(str(ROOT/'joint/detector/weights/best.pt'))
    records=[];start=time.monotonic()
    for n,target in enumerate(selected,1):
        key=target['image'];j=target['gt_index']
        ir=cv2.imread(str(BASE/f'llvip_yolo/infrared/images/val/{key}.jpg'),cv2.IMREAD_GRAYSCALE)
        c=cv2.imread(str(ROOT/f'joint/images/val/{key}.png'),cv2.IMREAD_GRAYSCALE)
        assert ir is not None and c is not None and ir.shape==c.shape
        ir=ir.astype(np.float32);c=c.astype(np.float32)
        blend=0.75*c+0.25*ir
        control=(c-c.mean())*(blend.std()/max(float(c.std()),1e-6))+blend.mean()
        h,w=c.shape
        gt=[]
        for line in (BASE/f'llvip_yolo/infrared/labels/val/{key}.txt').read_text().splitlines():
            _,x,y,bw,bh=map(float,line.split())
            gt.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        gt=np.array(gt).reshape(-1,4)
        for name,a in (('original',c),('blend_0.25IR',blend),('moment_control',control)):
            im=np.repeat(np.clip(np.rint(a),0,255).astype(np.uint8)[:,:,None],3,axis=2)
            t=time.monotonic()
            prediction=model.predict(im,device='cpu',imgsz=640,batch=1,
                rect=not args.matched_val_preprocess,
                conf=0.001,iou=0.7,max_det=300,half=False,verbose=False)[0]
            bb=prediction.boxes.xyxy.cpu().numpy();scores=prediction.boxes.conf.cpu().numpy()
            order=np.argsort(-scores);bb=bb[order];scores=scores[order]
            mat=ious(bb,gt);hits=match(mat[scores>=0.25],0.5)
            valid=mat[:,j]>=0.5
            records.append({'image':key,'target_gt':j,'group':target['group'],'variant':name,
                'target_hit':j in hits,'target_score_iou05':float(scores[valid].max()) if valid.any() else 0,
                'target_best_iou':float(mat[:,j].max()) if len(mat) else 0,
                'tp':len(hits),'fp':int((scores>=0.25).sum())-len(hits),'fn':len(gt)-len(hits),
                'input_mean':float(im.mean()/255),'input_std':float(im.std()/255),
                'seconds':time.monotonic()-t})
            with (dest/'records.jsonl').open('a') as f:f.write(json.dumps(records[-1])+'\n')
        print(json.dumps({'images':n,'total':24,'elapsed_seconds':time.monotonic()-start}),flush=True)
    summary={}
    for group in ('ir_only','C_only','both'):
        summary[group]={}
        for variant in ('original','blend_0.25IR','moment_control'):
            rr=[r for r in records if r['group']==group and r['variant']==variant]
            summary[group][variant]={'targets':len(rr),'hits':sum(r['target_hit'] for r in rr),
               'mean_target_score':float(np.mean([r['target_score_iou05'] for r in rr])),
               'mean_target_best_iou':float(np.mean([r['target_best_iou'] for r in rr]))}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    (dest/'complete.json').write_text(json.dumps({'predictions':len(records),'seconds':time.monotonic()-start}))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
