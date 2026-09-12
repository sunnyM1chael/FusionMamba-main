"""Prepare exact frozen infrared development lists and verify A labels."""
import hashlib
import json
from pathlib import Path

BASE = Path('/root/autodl-fs/research_protocol/v1')
RUN = BASE / 'runs/detection_ac_corrected_seed42_v1/full'
OUT = RUN / 'infrared'
OUT.mkdir(exist_ok=False)
digests = {'train': '7154252d9ce730dc9e50d925433084e1489bb535cca628b5f0f4c0d83b8ef5db',
           'val': 'd0da3502753369bdd3e3777ae2b23a68e61b71304c2e9b04881046ed109b30f0'}
reference = json.loads((RUN / 'none/export.json').read_text())
labels = {(r['split'], r['source']): r['label_sha256'] for r in reference['rows']}
counts = {}
for split, digest in digests.items():
    manifest = BASE / f'development_splits/llvip/{split}.txt'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == digest
    names = manifest.read_text().splitlines()
    paths = []
    for name in names:
        image = BASE / f'llvip_yolo/infrared/images/{split}/{name}'
        label = BASE / f'llvip_yolo/infrared/labels/{split}/{Path(name).stem}.txt'
        assert image.is_file()
        assert hashlib.sha256(label.read_bytes()).hexdigest() == labels[(split, name)]
        paths.append(str(image))
    (OUT / f'{split}.txt').write_text('\n'.join(paths) + '\n')
    counts[split] = len(paths)
assert counts == {'train': 10825, 'val': 1200}
(OUT / 'data.yaml').write_text(f'path: {OUT}\ntrain: train.txt\nval: val.txt\nnames:\n  0: person\n')
(OUT / 'export.json').write_text(json.dumps({'modality': 'infrared', 'counts': counts,
    'manifest_hashes': digests, 'labels_match_A': True, 'official_test_used': False,
    'policy': 'Original infrared JPEGs; loader expands to 3 channels; no re-encoding.'}, indent=2))
print('IR_PREPARATION_VERIFIED', counts, flush=True)
