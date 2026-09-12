"""Matched standard YOLO11n detector, strict health and initialization gates."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'yolo11_dsdam_deploy'))
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

ROOT = Path('/root/autodl-fs/research_protocol/v1/runs/detection_ac_corrected_seed42_v1')


class StrictTrainer(DetectionTrainer):
    def optimizer_step(self):
        if self.loss is not None and not torch.isfinite(self.loss).all():
            raise FloatingPointError('Nonfinite training loss')
        for parameter in self.model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('Nonfinite gradient')
        super().optimizer_step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('none', 'joint', 'infrared'), required=True)
    parser.add_argument('--stage', choices=('smoke', 'pilot', 'full'), required=True)
    args = parser.parse_args()
    weights = SOURCE / ('yolo11n.pt' if args.stage == 'smoke' else 'yolo11s_official.pt')
    expected = {'smoke': '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1',
                'pilot': '85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5',
                'full': '85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5'}
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == expected[args.stage]
    folder = ROOT / args.stage / args.mode
    data_folder = ROOT / ('smoke' if args.stage == 'smoke' else 'full') / args.mode
    assert (data_folder / 'export.json').is_file()
    folder.mkdir(parents=True, exist_ok=False) if folder != data_folder else None
    model = YOLO(str(weights))
    epoch_start = [None]

    def initialization(trainer):
        trainer._oom_retries = 3  # Fail instead of silently changing one group's batch size.
        digest = hashlib.sha256()
        for name, tensor in trainer.model.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        value = digest.hexdigest()
        reference = ROOT / args.stage / 'none' / 'initialization.json'
        if args.mode in ('joint', 'infrared'):
            assert json.loads(reference.read_text())['model_sha256'] == value, 'A/C initialization mismatch'
        (folder / 'initialization.json').write_text(json.dumps({'model_sha256': value}))

    def epoch_begin(trainer):
        epoch_start[0] = time.monotonic()

    def epoch_end(trainer):
        metrics = {k: float(v) for k, v in trainer.metrics.items()}
        assert all(math.isfinite(v) for v in metrics.values()), metrics
        record = {'epoch': trainer.epoch + 1, 'seconds_with_validation': time.monotonic() - epoch_start[0],
                  'metrics': metrics, 'batch': trainer.batch_size}
        with (folder / 'epoch_health.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')

    model.add_callback('on_pretrain_routine_end', initialization)
    model.add_callback('on_train_epoch_start', epoch_begin)
    model.add_callback('on_fit_epoch_end', epoch_end)
    model.train(trainer=StrictTrainer, data=str(data_folder / 'data.yaml'),
                epochs=3 if args.stage in {'smoke', 'pilot'} else 150, imgsz=640, batch=16,
                device=0, workers=4, seed=42, deterministic=True,
                optimizer='AdamW', lr0=0.001, lrf=0.01, cos_lr=True,
                patience=0, close_mosaic=0 if args.stage == 'smoke' else 10,
                amp=False, cache=False, plots=False, project=str(folder),
                name='detector', exist_ok=False)
    best = folder / 'detector/weights/best.pt'
    assert best.is_file()
    loaded = YOLO(str(best))
    assert all(torch.isfinite(v).all() for v in loaded.model.state_dict().values())
    (folder / 'training_complete.json').write_text(json.dumps({'best': str(best),
        'sha256': hashlib.sha256(best.read_bytes()).hexdigest(), 'stage': args.stage}))
    print('DETECTOR_COMPLETE', args.stage, args.mode, flush=True)


if __name__ == '__main__':
    main()
