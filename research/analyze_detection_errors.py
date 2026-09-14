"""Paired development diagnostics; fixed thresholds, not a replacement for AP."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

BASE = Path('/root/autodl-fs/research_protocol/v1')
ROOT = BASE / 'runs/detection_ac_corrected_seed42_v1/full'
OUT = BASE / 'results/detection_error_analysis_v1'


def ious(boxes, truth):
    if not len(boxes) or not len(truth):
        return np.zeros((len(boxes), len(truth)))
    lo = np.maximum(boxes[:, None, :2], truth[None, :, :2])
    hi = np.minimum(boxes[:, None, 2:], truth[None, :, 2:])
    inter = np.maximum(hi-lo, 0).prod(-1)
    a = (boxes[:, 2:]-boxes[:, :2]).prod(-1)
    b = (truth[:, 2:]-truth[:, :2]).prod(-1)
    return inter / np.maximum(a[:, None]+b[None, :]-inter, 1e-9)


def match(matrix, threshold):
    # Predictions already sorted by confidence; each GT is matched at most once.
    used = set()
    for row in matrix:
        available = [i for i in range(len(row)) if i not in used and row[i] >= threshold]
        if available:
            used.add(max(available, key=lambda i: row[i]))
    return used


def main():
    assert (OUT / 'complete.json').is_file()
    paths = [Path(p) for p in (ROOT / 'infrared/val.txt').read_text().splitlines()]
    gt = {}
    for image in paths:
        with Image.open(image) as im:
            w, h = im.size
        label = BASE / f'llvip_yolo/infrared/labels/val/{image.stem}.txt'
        boxes = []
        for line in label.read_text().splitlines():
            cls, x, y, bw, bh = map(float, line.split())
            assert cls == 0
            boxes.append([(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h])
        gt[image.stem] = np.array(boxes).reshape(-1, 4)
    assert len(gt) == 1200 and sum(map(len, gt.values())) == 3778
    predictions = {}
    for mode in ('infrared', 'joint', 'none', 'visible'):
        groups = defaultdict(list)
        for row in json.loads((OUT / mode / 'predictions.json').read_text()):
            key = Path(row['file_name']).stem
            assert key in gt and row['category_id'] == 1
            groups[key].append(row)
        predictions[mode] = groups
    report = {'scope': '1200 development images; confidence-greedy IoU matching, not AP or causal proof',
              'settings': [], 'cases_conf025_iou050': []}
    for conf in (0.1, 0.25, 0.5):
        for threshold in (0.5, 0.75):
            totals = {}
            matched = {}
            for mode, groups in predictions.items():
                tp = fp = fn = 0
                matched[mode] = {}
                for key, truth in gt.items():
                    rows = sorted((r for r in groups[key] if r['score'] >= conf),
                                  key=lambda r: -r['score'])
                    boxes = np.array([r['bbox'] for r in rows]).reshape(-1, 4)
                    boxes[:, 2:] += boxes[:, :2]
                    hits = match(ious(boxes, truth), threshold)
                    matched[mode][key] = hits
                    tp += len(hits)
                    fp += len(rows)-len(hits)
                    fn += len(truth)-len(hits)
                totals[mode] = {'tp': tp, 'fp': fp, 'fn': fn,
                                'precision': tp/max(tp+fp, 1), 'recall': tp/(tp+fn)}
            ir_only = c_only = both = neither = 0
            for key, truth in gt.items():
                ir, c = matched['infrared'][key], matched['joint'][key]
                ir_only += len(ir-c)
                c_only += len(c-ir)
                both += len(ir & c)
                neither += len(truth)-len(ir | c)
                if conf == 0.25 and threshold == 0.5 and ir != c:
                    report['cases_conf025_iou050'].append({'image': key,
                        'ir_only_gt': sorted(ir-c), 'fusion_only_gt': sorted(c-ir)})
            report['settings'].append({'confidence': conf, 'iou': threshold,
                'totals': totals, 'paired_ir_C': {'ir_only': ir_only, 'C_only': c_only,
                'both': both, 'neither': neither}})
    (OUT / 'paired_errors.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'settings': report['settings'],
                      'disagreement_images': len(report['cases_conf025_iou050'])}, indent=2))


if __name__ == '__main__':
    main()
