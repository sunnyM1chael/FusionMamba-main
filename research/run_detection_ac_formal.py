"""Release-gated YOLO11s pilot then serial A/C formal development training."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
ROOT = Path('/root/autodl-fs/research_protocol/v1/runs/detection_ac_corrected_seed42_v1')
WEIGHTS = SOURCE / 'yolo11s_official.pt'
WEIGHTS_SHA = '85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5'


def atomic_status(value):
    path = ROOT / 'formal_status.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def verify_export(mode):
    folder = ROOT / 'full' / mode
    report = json.loads((folder / 'export.json').read_text())
    assert len(report['rows']) == 12025
    assert {r['split'] for r in report['rows']} == {'train', 'val'}
    assert len(list((folder / 'images/train').glob('*.png'))) == 10825
    assert len(list((folder / 'images/val').glob('*.png'))) == 1200
    assert len(list((folder / 'labels/train').glob('*.txt'))) == 10825
    assert len(list((folder / 'labels/val').glob('*.txt'))) == 1200
    seen = set()
    protocol = ROOT.parents[1] / 'development_splits/llvip'
    expected = {(split, Path(name).stem + '.png')
                for split in ('train', 'val')
                for name in (protocol / f'{split}.txt').read_text().splitlines() if name.strip()}
    assert {(r['split'], r['output']) for r in report['rows']} == expected
    for index, row in enumerate(report['rows'], 1):
        key = (row['split'], row['output'])
        assert key not in seen
        seen.add(key)
        image = folder / 'images' / row['split'] / row['output']
        label = folder / 'labels' / row['split'] / f"{Path(row['output']).stem}.txt"
        assert hashlib.sha256(image.read_bytes()).hexdigest() == row['sha256']
        assert hashlib.sha256(label.read_bytes()).hexdigest() == row['label_sha256']
        original_label = ROOT.parents[1] / 'llvip_yolo/infrared/labels' / row['split'] / label.name
        assert hashlib.sha256(original_label.read_bytes()).hexdigest() == row['label_sha256']
        if index % 2000 == 0:
            print('VERIFY', mode, index, flush=True)
    print('EXPORT_VERIFIED', mode, len(seen), flush=True)


def run(mode, stage):
    atomic_status({'phase': stage, 'mode': mode})
    subprocess.run([sys.executable, str(SOURCE / 'research/train_detection_ac.py'),
                    '--mode', mode, '--stage', stage], check=True)


def main():
    if (ROOT / 'formal_status.json').exists():
        raise FileExistsError('Formal queue already exists; inspect it instead of duplicating runs.')
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == WEIGHTS_SHA
    assert shutil.disk_usage(ROOT).free > 20 * 2**30
    assert json.loads((ROOT / 'status.json').read_text())['phase'] == 'READY_FOR_FORMAL_REVIEW'
    for mode in ('none', 'joint'):
        verify_export(mode)
    reports = [json.loads((ROOT / 'full' / mode / 'export.json').read_text()) for mode in ('none', 'joint')]
    assert {(r['split'], r['source'], r['label_sha256']) for r in reports[0]['rows']} == {
        (r['split'], r['source'], r['label_sha256']) for r in reports[1]['rows']}
    metric_groups = Path('/root/autodl-fs/research_protocol/v1/results/llvip_detection_metric_groups_v1.json')
    assert metric_groups.is_file()
    run('none', 'pilot')
    # Pilot is a health/timing gate only. It never selects A versus C.
    run('none', 'full')
    run('joint', 'full')
    atomic_status({'phase': 'FORMAL_COMPLETE', 'runs': ['none', 'joint'],
                   'official_test_used': False})
    print('FORMAL_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
