"""Build versioned LLVIP YOLO views with symlinked images and audited labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from split_manifest import read_manifest, manifest_sha256


def convert_annotation(path):
    root = ET.parse(path).getroot()
    width, height = int(root.findtext('size/width')), int(root.findtext('size/height'))
    if width <= 0 or height <= 0:
        raise ValueError(f'Invalid XML image size in {path}')
    rows, rejected = [], []
    for index, obj in enumerate(root.findall('object')):
        if (obj.findtext('name') or '').strip().lower() != 'person':
            raise ValueError(f'Unexpected class in {path}')
        box = [float(obj.findtext('bndbox/'+key)) for key in ('xmin','ymin','xmax','ymax')]
        x1,y1,x2,y2 = box
        if x2 <= x1 or y2 <= y1:
            rejected.append({'object_index':index,'box':box,'reason':'non_positive_extent'})
            continue
        if min(x1,y1) < 0 or x2 > width or y2 > height:
            rejected.append({'object_index':index,'box':box,'reason':'out_of_bounds'})
            continue
        rows.append(f'0 {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} '
                    f'{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}')
    return rows, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llvip_root', type=Path, required=True)
    parser.add_argument('--protocol_dir', type=Path, required=True)
    parser.add_argument('--modality', choices=('infrared','visible'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rejected, counts = [], {}
    annotations = args.llvip_root/'Annotations'
    for split in ('train','val','test'):
        names = read_manifest(args.protocol_dir/(split+'.txt'))
        source_partition = 'test' if split == 'test' else 'train'
        source = args.llvip_root/args.modality/source_partition
        image_out, label_out = args.output/'images'/split, args.output/'labels'/split
        image_out.mkdir(parents=True)
        label_out.mkdir(parents=True)
        boxes = 0
        for name in names:
            image = source/name
            xml = annotations/(Path(name).stem+'.xml')
            if not image.is_file() or not xml.is_file():
                raise FileNotFoundError(f'Missing image or annotation for {name}')
            destination = image_out/name
            destination.symlink_to(image.resolve())
            rows, bad = convert_annotation(xml)
            (label_out/(Path(name).stem+'.txt')).write_text('\n'.join(rows)+('\n' if rows else ''),encoding='utf-8')
            boxes += len(rows)
            rejected.extend({'split':split,'name':name,**item} for item in bad)
        counts[split] = {'images':len(names),'boxes':boxes,
                         'manifest_sha256':manifest_sha256(args.protocol_dir/(split+'.txt'))}
    yaml = (f'path: {args.output.resolve().as_posix()}\ntrain: images/train\nval: images/val\n'
            'test: images/test\nnames:\n  0: person\n')
    (args.output/'LLVIP.yaml').write_text(yaml,encoding='utf-8')
    report = {'dataset':'LLVIP','modality':args.modality,'counts':counts,
              'rejected_boxes':rejected,
              'conversion':'VOC center/extent normalized by XML size; matches official LLVIP formula',
              'raw_annotations_modified':False}
    encoded = json.dumps(report,ensure_ascii=False,indent=2)
    (args.output/'conversion_report.json').write_text(encoded,encoding='utf-8')
    print(encoded)


if __name__ == '__main__':
    main()
