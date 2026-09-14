"""Descriptive score/localization audit; no causal or geometric-registration claims."""
import json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from PIL import Image
from analyze_detection_errors import BASE, ROOT, OUT, ious


def main():
    evidence=json.loads((OUT/'image_evidence_v1/objects.json').read_text())
    grouped=defaultdict(list)
    for row in evidence:
        grouped[row['image']].append(row)
    predictions={}
    for mode in ('infrared','joint'):
        predictions[mode]=defaultdict(list)
        for p in json.loads((OUT/mode/'predictions.json').read_text()):
            predictions[mode][Path(p['file_name']).stem].append(p)
    objects=[]
    for key, ev in grouped.items():
        path=BASE/f'llvip_yolo/infrared/images/val/{key}.jpg'
        with Image.open(path) as im:
            w,h=im.size
        labels=[list(map(float,line.split())) for line in
                (BASE/f'llvip_yolo/infrared/labels/val/{key}.txt').read_text().splitlines()]
        boxes=np.array([[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
                        for _,x,y,bw,bh in labels]).reshape(-1,4)
        metrics={}
        for mode in predictions:
            pp=predictions[mode][key]
            bb=np.array([p['bbox'] for p in pp]).reshape(-1,4)
            bb[:,2:]+=bb[:,:2]
            scores=np.array([p['score'] for p in pp])
            mat=ious(bb,boxes)
            metrics[mode]=[]
            for j in range(len(boxes)):
                best=float(mat[:,j].max()) if len(pp) else 0
                good=mat[:,j]>=0.5
                score=float(scores[good].max()) if good.any() else 0
                confident=scores>=0.25
                biou=float(mat[confident,j].max()) if confident.any() else 0
                metrics[mode].append({'best_iou_any_score':best,
                    'best_score_iou05':score,'best_iou_conf025':biou})
        for e in ev:
            j=e['gt_index']; box=boxes[j]
            row={'image':key,'gt_index':j,'group':e['group'],
                 'prefix':key[:2],'touches_image_border':bool(box[0]<=2 or box[1]<=2 or box[2]>=w-2 or box[3]>=h-2),
                 'area_resized640':float((box[2]-box[0])*(box[3]-box[1])*(640/max(w,h))**2),
                 'infrared':metrics['infrared'][j],'joint':metrics['joint'][j]}
            losing='joint' if e['group']=='ir_only' else 'infrared'
            if e['group'] in ('ir_only','C_only'):
                m=metrics[losing][j]
                row['losing_detector_diagnostic']=('match_competition' if m['best_score_iou05']>=0.25 else
                   'below_confidence_threshold' if m['best_score_iou05']>0 else
                   'localization_below_iou05' if m['best_iou_any_score']>=0.1 else 'no_overlapping_proposal')
            objects.append(row)
    report={}
    for group in ('both','ir_only','C_only','neither'):
        rr=[r for r in objects if r['group']==group]
        report[group]={'n':len(rr),'border_count':sum(r['touches_image_border'] for r in rr),
           'median_area640':float(np.median([r['area_resized640'] for r in rr])),
           'prefix_counts':dict(Counter(r['prefix'] for r in rr)),
           'failure_types':dict(Counter(r.get('losing_detector_diagnostic','NA') for r in rr))}
        for mode in predictions:
            report[group][mode]={k:float(np.mean([r[mode][k] for r in rr])) for k in metrics[mode][0]}
    dest=OUT/'failure_mechanisms_v1'
    dest.mkdir(exist_ok=False)
    (dest/'objects.json').write_text(json.dumps(objects))
    (dest/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
