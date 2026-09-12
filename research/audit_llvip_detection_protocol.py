"""Freeze LLVIP validation box-size groups before detector results are viewed."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path('/root/autodl-fs/research_protocol/v1')
DATA = Path('/root/autodl-fs/datasets/LLVIP')
OUT = ROOT / 'results/llvip_detection_metric_groups_v1.json'
if OUT.exists():
    raise FileExistsError(OUT)

manifest = ROOT / 'development_splits/llvip/val.txt'
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == 'd0da3502753369bdd3e3777ae2b23a68e61b71304c2e9b04881046ed109b30f0'
names = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
counts = Counter()
areas = []
for name in names:
    with Image.open(DATA / 'infrared/train' / name) as image:
        width, height = image.size
    gain = min(640 / width, 640 / height)
    label = ROOT / f'llvip_yolo/infrared/labels/val/{Path(name).stem}.txt'
    for line in label.read_text().splitlines():
        _, _, _, relative_width, relative_height = map(float, line.split())
        area = relative_width * width * relative_height * height
        areas.append(area)
        counts['all'] += 1
        counts['native_lt_32sq'] += area < 32**2
        counts['detector640_lt_32sq'] += area * gain**2 < 32**2

report = {
    'status': 'frozen_before_detector_results',
    'validation_images': len(names),
    'boxes': dict(counts),
    'definitions': {
        'native_lt_32sq': 'ground-truth box area < 32^2 pixels in the native 1280x1024 image',
        'detector640_lt_32sq': 'ground-truth box area after isotropic letterbox gain to 640 < 32^2 pixels'},
    'primary_metric': 'validation AP50:95 over all boxes',
    'secondary_metrics': ['AP50', 'precision', 'recall'],
    'small_object_metrics': 'report both predeclared groups with instance counts; do not choose a definition by observed AP',
    'limitations': ['Internal LLVIP validation is not verified scene-disjoint.',
                    'Size-stratified AP implementation must be unit-tested before reporting.']}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
