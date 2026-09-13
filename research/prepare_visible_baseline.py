"""Reuse original visible images and verified labels on the frozen development split."""
import hashlib
import json
from pathlib import Path

BASE = Path('/root/autodl-fs/research_protocol/v1')
RUN = BASE / 'runs/detection_ac_corrected_seed42_v1/full'
OUT = RUN / 'visible'
VIEW = Path('/root/autodl-tmp/llvip_visible_seed42_view')
RAW = Path('/root/autodl-fs/datasets/LLVIP/visible/train')
DIGESTS = {'train': '7154252d9ce730dc9e50d925433084e1489bb535cca628b5f0f4c0d83b8ef5db',
           'val': 'd0da3502753369bdd3e3777ae2b23a68e61b71304c2e9b04881046ed109b30f0'}


def main():
    assert not OUT.exists(), f'Existing run: {OUT}'
    reference = json.loads((RUN / 'none/export.json').read_text())
    labels = {(r['split'], r['source']): r['label_sha256'] for r in reference['rows']}
    paths = {}
    for split, digest in DIGESTS.items():
        manifest = BASE / f'development_splits/llvip/{split}.txt'
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == digest
        paths[split] = []
        for name in manifest.read_text().splitlines():
            assert Path(name).name == name
            image = RAW / name
            label = BASE / f'llvip_yolo/infrared/labels/{split}/{Path(name).stem}.txt'
            assert image.is_file(), image
            assert hashlib.sha256(label.read_bytes()).hexdigest() == labels[(split, name)]
            paths[split].append((image, label))
    assert {k: len(v) for k, v in paths.items()} == {'train': 10825, 'val': 1200}
    assert not ({p.name for p, _ in paths['train']} & {p.name for p, _ in paths['val']})
    VIEW.mkdir(exist_ok=False)
    OUT.mkdir(exist_ok=False)
    for split, pairs in paths.items():
        images = VIEW / 'images' / split
        images.mkdir(parents=True)
        label_dir = VIEW / 'labels' / split
        label_dir.parent.mkdir(exist_ok=True)
        label_dir.symlink_to(BASE / f'llvip_yolo/infrared/labels/{split}', target_is_directory=True)
        for image, _ in pairs:
            (images / image.name).symlink_to(image)
        (OUT / f'{split}.txt').write_text('\n'.join(str(images / p.name) for p, _ in pairs) + '\n')
    (OUT / 'data.yaml').write_text(f'path: {OUT}\ntrain: train.txt\nval: val.txt\nnames:\n  0: person\n')
    (OUT / 'export.json').write_text(json.dumps({'modality': 'visible',
        'counts': {k: len(v) for k, v in paths.items()}, 'manifest_hashes': DIGESTS,
        'labels_match_A': True, 'official_test_used': False,
        'policy': 'Original visible JPEGs, no re-encoding; symlinks on local server disk.'}, indent=2))
    print('VISIBLE_PREPARATION_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
