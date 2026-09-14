"""IR-anchored strong-edge retention proxy, not segmentation or registration truth."""
import json
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw
from analyze_detection_errors import BASE,ROOT,OUT


def main():
    rows=json.loads((OUT/'failure_mechanisms_v1/objects.json').read_text())
    groups=defaultdict(list)
    for r in rows: groups[r['image']].append(r)
    dest=OUT/'edge_check_v1';dest.mkdir(exist_ok=False)
    result=[];selected=set();shapes=set()
    for n,(key,rr) in enumerate(groups.items(),1):
        ir=np.asarray(Image.open(BASE/f'llvip_yolo/infrared/images/val/{key}.jpg').convert('L'),dtype=np.float32)/255
        c=np.asarray(Image.open(ROOT/f'joint/images/val/{key}.png').convert('L'),dtype=np.float32)/255
        shapes.add(ir.shape);h,w=ir.shape
        labels=[list(map(float,s.split())) for s in (BASE/f'llvip_yolo/infrared/labels/val/{key}.txt').read_text().splitlines()]
        for r in rr:
            _,x,y,bw,bh=labels[r['gt_index']]
            l,t=max(0,int((x-bw/2)*w)),max(0,int((y-bh/2)*h))
            right,b=min(w,int(np.ceil((x+bw/2)*w))),min(h,int(np.ceil((y+bh/2)*h)))
            a=ir[t:b,l:right];f=c[t:b,l:right]
            if min(a.shape)<3: continue
            gy,gx=np.gradient(a);fy,fx=np.gradient(f)
            mag=np.hypot(gx,gy);fm=np.hypot(fx,fy)
            mask=(mag>=np.quantile(mag,0.9)) & (mag>0.01)
            if mask.any():
                cosine=(gx*fx+gy*fy)/np.maximum(mag*fm,1e-8)
                result.append({'image':key,'group':r['group'],
                    'mean_IR_edge':float(mag[mask].mean()),'mean_C_at_IR_edge':float(fm[mask].mean()),
                    'orientation_cosine':float(cosine[mask].mean()),
                    'edge_retained_fraction':float((fm[mask]>=0.5*mag[mask]).mean())})
            token=(r['group'],key[:2])
            if r['group'] in ('ir_only','C_only') and token not in selected:
                selected.add(token)
                pad=30;crop=(max(0,l-pad),max(0,t-pad),min(w,right+pad),min(h,b+pad))
                vis=np.asarray(Image.open('/root/autodl-fs/datasets/LLVIP/visible/train/'+key+'.jpg').convert('L'))
                panel=Image.new('RGB',(900,350),'white')
                for k,(title,arr) in enumerate((('IR',ir*255),('VIS',vis),('FUSED',c*255))):
                    im=Image.fromarray(arr.astype('uint8')).convert('RGB').crop(crop)
                    im.thumbnail((300,310));panel.paste(im,(300*k,35))
                    ImageDraw.Draw(panel).text((300*k+4,5),f'{title} {key} {r["group"]}',fill='black')
                panel.save(dest/f'{r["group"]}_{key[:2]}.jpg')
        if n%200==0: print('IMAGES',n,flush=True)
    summary={'shapes':sorted(shapes),'note':'Top10% IR gradients in GT rectangles, not anatomical contours or geometric truth.', 'groups':{}}
    for g in ('both','ir_only','C_only','neither'):
        rr=[r for r in result if r['group']==g]
        summary['groups'][g]={'n':len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in ('mean_IR_edge','mean_C_at_IR_edge','orientation_cosine','edge_retained_fraction')}}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    (dest/'objects.json').write_text(json.dumps(result))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
