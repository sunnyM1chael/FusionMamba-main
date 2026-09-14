"""Re-evaluate frozen best detectors and retain predictions for paired error analysis."""
import json
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'yolo11_dsdam_deploy'))
from ultralytics import YOLO

ROOT = Path('/root/autodl-fs/research_protocol/v1/runs/detection_ac_corrected_seed42_v1/full')
OUT = Path('/root/autodl-fs/research_protocol/v1/results/detection_error_analysis_v1')

if __name__ == '__main__':
    OUT.mkdir(exist_ok=False)
    summary = {}
    for mode in ('infrared', 'joint', 'none', 'visible'):
        folder = ROOT / mode
        assert (folder / 'training_complete.json').is_file()
        model = YOLO(str(folder / 'detector/weights/best.pt'))
        result = model.val(data=str(folder / 'data.yaml'), split='val', imgsz=640,
                           batch=16, device=0, workers=4, half=False,
                           conf=0.001, iou=0.7, max_det=300, augment=False,
                           save_json=True, plots=False, project=str(OUT), name=mode)
        summary[mode] = {k: float(v) for k, v in result.results_dict.items()}
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    (OUT / 'complete.json').write_text(json.dumps({'modes': list(summary),
        'scope': 'Frozen internal validation only; predictions for later paired error analysis.'}))
