"""Read image headers and verify paired/XML dimensions without decoding pixels."""
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT=Path('/root/autodl-fs/datasets/M3FD_Detection')
OUT=Path('/root/autodl-fs/research_protocol/v1/results/m3fd_pair_audit_v1.json')
EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}


def index(folder):return {p.stem:p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXT}


def main():
    ir,vis=index(ROOT/'Ir'),index(ROOT/'Vis')
    ann={p.stem:p for p in (ROOT/'Annotation').glob('*.xml')}
    bad=[];formats=Counter();modes=Counter();sizes=Counter()
    for n,key in enumerate(sorted(ann),1):
        root=ET.parse(ann[key]).getroot();s=root.find('size')
        xml=(int(s.findtext('width')),int(s.findtext('height')))
        with Image.open(ir[key]) as a:
            ai=a.size;formats[f'IR:{a.format}']+=1;modes[f'IR:{a.mode}']+=1
        with Image.open(vis[key]) as b:
            bi=b.size;formats[f'VIS:{b.format}']+=1;modes[f'VIS:{b.mode}']+=1
        sizes[f'{ai[0]}x{ai[1]}']+=1
        if not (ai==bi==xml):bad.append({'image':key,'ir':ai,'visible':bi,'xml':xml})
        if n%500==0:print('HEADERS',n,flush=True)
    result={'pairs':len(ann),'dimension_mismatches':bad,'formats':dict(formats),
       'modes':dict(modes),'sizes':dict(sizes),
       'limitations':['Header-level integrity only; pixel registration and exact/near duplicates are not assessed.',
                      'This audit does not define train/validation/test membership.']}
    OUT.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
