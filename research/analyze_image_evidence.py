"""CPU box-region contrast/detail proxies for all frozen validation targets."""
import json
import time
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw
from analyze_detection_errors import BASE, ROOT, OUT, ious, match


def measure(a, box, all_boxes):
    h, w = a.shape
    x1,y1,x2,y2 = box
    x1,y1 = max(0,int(x1)), max(0,int(y1))
    x2,y2 = min(w,int(np.ceil(x2))), min(h,int(np.ceil(y2)))
    pad = max(3, int(0.15*max(x2-x1,y2-y1)))
    l,t,r,b = max(0,x1-pad),max(0,y1-pad),min(w,x2+pad),min(h,y2+pad)
    region = a[t:b,l:r]
    mask = np.ones(region.shape, dtype=bool)
    for bx in all_boxes:
        lx,ty,rx,by = bx
        lx,ty = max(l,int(lx)),max(t,int(ty))
        rx,by = min(r,int(np.ceil(rx))),min(b,int(np.ceil(by)))
        if rx>lx and by>ty:
            mask[ty-t:by-t,lx-l:rx-l] = False
    fg = a[y1:y2,x1:x2]
    bg = region[mask]
    if not fg.size or len(bg)<10:
        return None
    edge = (np.abs(np.diff(fg,axis=0)).mean()+np.abs(np.diff(fg,axis=1)).mean())/2 if min(fg.shape)>1 else 0
    return {'absolute_contrast': float(abs(fg.mean()-bg.mean())),
            'signed_contrast': float(fg.mean()-bg.mean()),
            'foreground_gradient': float(edge), 'foreground_std': float(fg.std())}


def main():
    start=time.monotonic()
    dest=OUT/'image_evidence_v1'
    dest.mkdir(exist_ok=False)
    pred={}
    for mode in ('infrared','joint'):
        groups=defaultdict(list)
        for p in json.loads((OUT/mode/'predictions.json').read_text()):
            if p['score']>=0.25:
                groups[p['file_name'].rsplit('.',1)[0]].append(p)
        pred[mode]=groups
    paths=(ROOT/'infrared/val.txt').read_text().splitlines()
    rows=[]
    panels=defaultdict(int)
    for n,path in enumerate(paths,1):
        from pathlib import Path
        key=Path(path).stem
        ir=np.asarray(Image.open(path).convert('L'),dtype=np.float32)/255
        c=np.asarray(Image.open(ROOT/f'joint/images/val/{key}.png').convert('L'),dtype=np.float32)/255
        assert ir.shape==c.shape
        h,w=ir.shape
        boxes=[]
        for line in (BASE/f'llvip_yolo/infrared/labels/val/{key}.txt').read_text().splitlines():
            cls,x,y,bw,bh=map(float,line.split())
            boxes.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        boxes=np.array(boxes).reshape(-1,4)
        hits={}
        for mode in pred:
            pp=sorted(pred[mode][key],key=lambda v:-v['score'])
            bb=np.array([p['bbox'] for p in pp]).reshape(-1,4)
            bb[:,2:]+=bb[:,:2]
            hits[mode]=match(ious(bb,boxes),0.5)
        for j,box in enumerate(boxes):
            group=('both' if j in hits['infrared'] and j in hits['joint'] else
                   'ir_only' if j in hits['infrared'] else
                   'C_only' if j in hits['joint'] else 'neither')
            rows.append({'image':key,'gt_index':j,'group':group,
                         'infrared':measure(ir,box,boxes),'joint':measure(c,box,boxes)})
            if group in ('ir_only','C_only') and panels[group]<4:
                canvas=Image.new('RGB',(768,410),'white')
                for k,(label,a) in enumerate((('IR',ir),('SADFFM',c))):
                    im=Image.fromarray((a*255).astype('uint8')).convert('RGB')
                    ImageDraw.Draw(im).rectangle(tuple(box),outline='red',width=4)
                    im.thumbnail((384,380))
                    canvas.paste(im,(384*k,25))
                    ImageDraw.Draw(canvas).text((384*k+5,5),f'{label} {key} GT{j} {group}',fill='black')
                canvas.save(dest/f'{group}_{panels[group]:02d}.jpg')
                panels[group]+=1
        if n%50==0:
            progress={'images':n,'total':len(paths),'seconds':time.monotonic()-start}
            (dest/'progress.json').write_text(json.dumps(progress))
            print(json.dumps(progress),flush=True)
    (dest/'objects.json').write_text(json.dumps(rows))
    summary={}
    for group in ('both','ir_only','C_only','neither'):
        rr=[r for r in rows if r['group']==group and r['infrared'] and r['joint']]
        summary[group]={'valid_objects':len(rr)}
        for metric in ('absolute_contrast','foreground_gradient','foreground_std'):
            d=np.array([r['joint'][metric]-r['infrared'][metric] for r in rr])
            summary[group][metric]={'mean_C_minus_IR':float(d.mean()) if len(d) else None,
                'fraction_C_lower':float((d<0).mean()) if len(d) else None}
    (dest/'summary.json').write_text(json.dumps({'groups':summary,'objects':len(rows),
       'seconds':time.monotonic()-start,'limitations':'Box proxies, not masks or alignment ground truth; association only. Cases are first four per direction in frozen manifest order.'},indent=2))
    print('IMAGE_EVIDENCE_COMPLETE',flush=True)


if __name__=='__main__':
    main()
