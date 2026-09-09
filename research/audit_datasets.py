"""Read-only image/annotation audit. Outputs are new, versioned artifacts.

Run from the project root: python research/audit_datasets.py --output PATH
No samples are repartitioned, relabelled or deleted by this tool.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from PIL import Image
import numpy as np

EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def image_info(path):
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        rgb = image.convert('RGB')
        size = image.size
        pixels = hashlib.sha256(str(size).encode() + rgb.tobytes()).hexdigest()
        small = np.array(image.convert('L').resize((9, 8)))
        dhash = np.packbits(small[:, 1:] > small[:, :-1]).tobytes().hex()
    return {'width': size[0], 'height': size[1], 'sha256': hashlib.sha256(data).hexdigest(),
            'pixel_sha256': pixels, 'dhash': dhash}


def inspect_sample(job):
    dataset, partition, name, ir, vis, annotation = job
    row = {'dataset': dataset, 'partition': partition, 'name': name,
           'ir_path': str(ir), 'vis_path': str(vis), 'issues': []}
    try:
        row['ir'] = image_info(ir)
        row['vis'] = image_info(vis)
        size = (row['ir']['width'], row['ir']['height'])
        if size != (row['vis']['width'], row['vis']['height']):
            row['issues'].append('modality_dimension_mismatch')
        if annotation is not None:
            data = annotation.read_bytes()
            row['annotation_sha256'] = hashlib.sha256(data).hexdigest()
            root = ET.fromstring(data)
            row['xml_size'] = [root.findtext('size/width'), root.findtext('size/height')]
            if all(row['xml_size']) and tuple(map(int, row['xml_size'])) != size:
                row['issues'].append('xml_image_dimension_mismatch')
            objects = []
            for obj in root.findall('object'):
                box = [float(obj.findtext('bndbox/' + k)) for k in ('xmin','ymin','xmax','ymax')]
                x1,y1,x2,y2 = box
                if not all(np.isfinite(box)) or x2 <= x1 or y2 <= y1:
                    row['issues'].append('invalid_box')
                if min(x1,y1) < 0 or x2 > size[0] or y2 > size[1]:
                    row['issues'].append('out_of_bounds_box')
                objects.append({'class': obj.findtext('name'), 'box': box,
                                'difficult': obj.findtext('difficult', default='0')})
            row['objects'] = objects
    except Exception as exc:
        row['issues'].append(f'{type(exc).__name__}: {exc}')
    return row


def discover(root):
    layouts = [('MSRS', part, root/'MSRS'/part/'ir', root/'MSRS'/part/'vi', None)
               for part in ('train','test')]
    layouts += [('LLVIP', part, root/'LLVIP'/'infrared'/part, root/'LLVIP'/'visible'/part,
                 root/'LLVIP'/'Annotations') for part in ('train','test')]
    layouts += [('M3FD', 'unassigned', root/'M3FD_RAW'/'Ir', root/'M3FD_RAW'/'Vis',
                 root/'M3FD_RAW'/'Annotation'),
                ('M3FD_Fusion300', 'unassigned', root/'M3FD_Fusion'/'Ir', root/'M3FD_Fusion'/'Vis', None)]
    jobs, inventory = [], []
    for dataset, partition, ir, vis, ann in layouts:
        if not ir.is_dir() or not vis.is_dir():
            inventory.append({'dataset':dataset, 'partition':partition, 'error':'missing_directory'})
            continue
        a = {p.name:p for p in ir.iterdir() if p.suffix.lower() in EXT and p.is_file()}
        b = {p.name:p for p in vis.iterdir() if p.suffix.lower() in EXT and p.is_file()}
        inventory.append({'dataset':dataset,'partition':partition,'ir':len(a),'vis':len(b),
                          'ir_only':sorted(set(a)-set(b)), 'vis_only':sorted(set(b)-set(a))})
        for name in sorted(set(a)|set(b)):
            jobs.append((dataset, partition, name, ir/name, vis/name,
                         ann/(Path(name).stem+'.xml') if ann else None))
    return jobs, inventory


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/root/autodl-fs/datasets'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs, inventory = discover(args.root)
    counts, issues, classes, sizes = Counter(), [], Counter(), Counter()
    empty = []
    exact, perceptual = defaultdict(list), defaultdict(list)
    annotation_hash = hashlib.sha256()
    started = time.time()
    with (args.output/'samples.jsonl').open('w', encoding='utf-8') as f, ThreadPoolExecutor(args.workers) as pool:
        for i, row in enumerate(pool.map(inspect_sample, jobs), 1):
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
            key = row['dataset']+'/'+row['partition']
            counts[key] += 1
            sample = key+'/'+row['name']
            if row['issues']:
                issues.append({'sample':sample, 'issues':row['issues']})
            if 'objects' in row:
                classes.update(row['dataset']+'/'+o['class'] for o in row['objects'])
                if not row['objects']:
                    empty.append(sample)
                annotation_hash.update((sample+row['annotation_sha256']).encode())
            if 'ir' in row and 'vis' in row:
                sizes[(row['dataset'],row['ir']['width'],row['ir']['height'])] += 1
                exact[(row['ir']['pixel_sha256'],row['vis']['pixel_sha256'])].append(sample)
                perceptual[(row['ir']['dhash'],row['vis']['dhash'])].append(sample)
            if i % 500 == 0:
                print(f'audited {i}/{len(jobs)} ({time.time()-started:.0f}s)', flush=True)
    summary = {'inventory':inventory, 'counts':dict(counts), 'issues':issues, 'empty_annotations':empty,
               'classes':dict(classes), 'image_sizes':[{'dataset':k[0],'width':k[1],'height':k[2],'count':v} for k,v in sizes.items()],
               'exact_duplicate_pairs':[v for v in exact.values() if len(v)>1],
               'same_dhash_pair_candidates':[v for v in perceptual.values() if len(v)>1],
               'annotation_bundle_sha256':annotation_hash.hexdigest(),
               'limitations':['dHash equality is a candidate screen, not a full near-duplicate or scene audit',
                             'annotation release version and coordinate convention require source verification',
                             'no test images may be used for training; no partition created by this audit'],
               'elapsed_seconds':round(time.time()-started,2)}
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'counts':dict(counts),'issue_samples':len(issues),'empty':empty,
                      'exact_duplicate_groups':len(summary['exact_duplicate_pairs'])}), flush=True)


if __name__ == '__main__':
    main()
