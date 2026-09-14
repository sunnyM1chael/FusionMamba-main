"""Read-only M3FD annotation inventory for detection-side ablation planning."""
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path('/root/autodl-fs/datasets/M3FD_Detection')
OUT=Path('/root/autodl-fs/research_protocol/v1/results/m3fd_small_object_audit_v1.json')
EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}


def images(folder):
    return {p.stem:p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXT}


def main():
    ann={p.stem:p for p in (ROOT/'Annotation').glob('*.xml')}
    ir,vis=images(ROOT/'Ir'),images(ROOT/'Vis')
    keys=sorted(set(ann)&set(ir)&set(vis))
    result={'counts':{'annotations':len(ann),'ir':len(ir),'visible':len(vis),'triples':len(keys)},
            'missing':{'annotation':sorted((set(ir)&set(vis))-set(ann)),
                       'infrared':sorted((set(ann)&set(vis))-set(ir)),
                       'visible':sorted((set(ann)&set(ir))-set(vis))},
            'classes':{},'image_sizes':{},'invalid_boxes':[],'empty_images':0,
            'boxes_total':0,'resized640_area_lt_32sq':0,'resized640_area_lt_16sq':0,
            'per_class':{},'policy':'Inventory only; no train/val/test split chosen.'}
    classes=Counter();sizes=Counter();per=defaultdict(lambda:Counter(total=0,small32=0,small16=0))
    for key in keys:
        root=ET.parse(ann[key]).getroot()
        size=root.find('size');w=int(size.findtext('width'));h=int(size.findtext('height'))
        sizes[f'{w}x{h}']+=1;n=0;scale=640/max(w,h)
        for obj in root.findall('object'):
            name=obj.findtext('name').strip();b=obj.find('bndbox')
            x1,y1,x2,y2=[float(b.findtext(v)) for v in ('xmin','ymin','xmax','ymax')]
            if not (0<=x1<x2<=w and 0<=y1<y2<=h):
                result['invalid_boxes'].append({'image':key,'class':name,'box':[x1,y1,x2,y2],'size':[w,h]})
                continue
            area=(x2-x1)*(y2-y1)*scale*scale
            classes[name]+=1;per[name]['total']+=1;n+=1;result['boxes_total']+=1
            if area<32**2:result['resized640_area_lt_32sq']+=1;per[name]['small32']+=1
            if area<16**2:result['resized640_area_lt_16sq']+=1;per[name]['small16']+=1
        if n==0:result['empty_images']+=1
    result['classes']=dict(classes);result['image_sizes']=dict(sizes)
    result['per_class']={k:dict(v) for k,v in per.items()}
    result['notes']=['Small counts use box area after aspect-preserving long-side resize to640, before padding.',
       'Thresholds are planning proxies, not an official M3FD metric or proof of scene-independent splits.',
       'XML validity uses declared size; paired pixel dimensions and duplicates require separate audit.']
    OUT.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
