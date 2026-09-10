#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""Reproducible FusionMamba training with validation and resumable checkpoints."""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from logger import setup_logger
from loss import Fusionloss
from TaskFusion_dataset import Fusion_dataset
from split_manifest import read_manifest, manifest_sha256


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(dataset, batch_size, workers, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= batch_size,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def compute_loss(model, criterion, image_vis, image_ir, device, amp, amp_dtype=torch.float16):
    image_vis = image_vis.to(device, non_blocking=True)
    image_ir = image_ir.to(device, non_blocking=True)
    height, width = image_ir.shape[-2:]
    pad_h, pad_w = (-height) % 32, (-width) % 32
    model_vis = F.pad(image_vis, (0, pad_w, 0, pad_h), mode='replicate') if pad_h or pad_w else image_vis
    model_ir = F.pad(image_ir, (0, pad_w, 0, pad_h), mode='replicate') if pad_h or pad_w else image_ir
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
        fusion_image = model(model_ir, model_vis)[..., :height, :width]
        total, intensity, ssim, gradient = criterion(
            image_vis=image_vis,
            image_ir=image_ir,
            generate_img=fusion_image,
            i=0,
            labels=None,
        )
    return total, intensity, ssim, gradient


@torch.no_grad()
def validate(model, loader, criterion, device, amp, amp_dtype=torch.float16):
    model.eval()
    totals = np.zeros(4, dtype=np.float64)
    for image_vis, image_ir in loader:
        losses = compute_loss(model, criterion, image_vis, image_ir, device, amp, amp_dtype)
        totals += np.asarray([loss.item() for loss in losses])
    return totals / max(len(loader), 1)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val, history, args, loaders=()):
    torch.save(
        {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val': best_val,
            'history': history,
            'args': vars(args),
            'rng': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                'loaders': [loader.generator.get_state() for loader in loaders],
            },
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, scaler, device, loaders=()):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if 'model' not in checkpoint:
        model.load_state_dict(checkpoint)
        return 0, float('inf'), []
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    if checkpoint.get('scaler'):
        scaler.load_state_dict(checkpoint['scaler'])
    if 'rng' in checkpoint:
        rng = checkpoint['rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'].cpu())
        if rng['cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])
        if len(loaders) != len(rng['loaders']):
            raise ValueError('Checkpoint data-loader count differs from this run')
        for loader, state in zip(loaders, rng['loaders']):
            loader.generator.set_state(state.cpu())
    else:
        logging.warning('Legacy checkpoint has no RNG state; continuation is not sequence-equivalent.')
    return (
        int(checkpoint.get('epoch', -1)) + 1,
        float(checkpoint.get('best_val', float('inf'))),
        checkpoint.get('history', []),
    )


def load_pretrained(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)


def plot_history(history, output_path):
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [item['epoch'] for item in history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, key, title in zip(
        axes.flat,
        ('total', 'intensity', 'gradient', 'ssim'),
        ('Total loss', 'Intensity loss', 'Gradient loss', 'SSIM loss'),
    ):
        axis.plot(epochs, [item[f'train_{key}'] for item in history], label='train')
        valid = [item.get(f'val_{key}') for item in history]
        if all(value is not None for value in valid):
            axis.plot(epochs, valid, label='val')
        axis.set_title(title)
        axis.set_xlabel('epoch')
        axis.grid(True)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def train_fusion(args, logger):
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Restore a GPU instance or explicitly request --device cpu.')
    if not args.kaist_root and (not args.ir_path or not args.vis_path):
        raise ValueError('Provide --ir_path/--vis_path or --kaist_root')
    if args.resume and args.pretrained:
        raise ValueError('Use only one of --resume and --pretrained')
    if args.train_list and args.val_list:
        overlap = set(read_manifest(args.train_list)) & set(read_manifest(args.val_list))
        if overlap:
            raise ValueError(f'Train/validation overlap: {sorted(overlap)[:5]}')
    from models.vmamba_Fusion_efficross import VSSM_Fusion

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {name: {'path': str(Path(path).resolve()), 'sha256': manifest_sha256(path)}
                  for name, path in [('train', args.train_list), ('val', args.val_list)] if path}
    (output_dir / 'split_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    with (output_dir / 'train_args.json').open('w', encoding='utf-8') as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    if torch.cuda.is_available() and args.device != 'cpu':
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    amp = args.amp and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16

    model = VSSM_Fusion(
        use_dsdam=not args.disable_dsdam,
        share_encoder_weights=args.share_encoder_weights,
        weighting_mode=args.weighting_mode,
    ).to(device)
    criterion = Fusionloss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr
    )
    scaler_enabled = amp and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=scaler_enabled)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    if args.pretrained:
        load_pretrained(args.pretrained, model, device)
        logger.info('Loaded model weights from %s without optimizer state', args.pretrained)

    train_set = Fusion_dataset(
        'train', args.ir_path, args.vis_path, args.length, args.crop_size, args.train_list, args.kaist_root
    )
    train_loader = build_loader(
        train_set, args.batch_size, args.num_workers, True, args.seed
    )

    val_loader = None
    if args.val_list:
        val_set = Fusion_dataset(
            'val', args.ir_path, args.vis_path, 0, args.crop_size, args.val_list, args.kaist_root
        )
        val_loader = build_loader(val_set, 1, args.num_workers, False, args.seed)

    start_epoch, best_val, history = 0, float('inf'), []
    if args.resume:
        start_epoch, best_val, history = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device,
            [loader for loader in (train_loader, val_loader) if loader is not None]
        )
        logger.info('Resumed %s at epoch %d', args.resume, start_epoch)

    logger.info(
        'Start: device=%s amp=%s train=%d val=%d SACAFM=%s',
        device,
        amp,
        len(train_set),
        len(val_loader.dataset) if val_loader else 0,
        not args.disable_dsdam,
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        sums = np.zeros(4, dtype=np.float64)
        started = time.time()
        optimizer_updates = 0
        skipped_updates = 0
        for step, (image_vis, image_ir) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            losses = compute_loss(model, criterion, image_vis, image_ir, device, amp, amp_dtype)
            if not torch.isfinite(losses[0]):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch + 1}, step {step}')
            scaler.scale(losses[0]).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=not amp)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < scale_before:
                skipped_updates += 1
            else:
                optimizer_updates += 1
            sums += np.asarray([loss.item() for loss in losses])
            if step % args.log_interval == 0:
                logger.info(
                    'epoch %d/%d step %d/%d loss %.5f lr %.2e',
                    epoch + 1,
                    args.epochs,
                    step,
                    len(train_loader),
                    losses[0].item(),
                    optimizer.param_groups[0]['lr'],
                )

        train_values = sums / max(len(train_loader), 1)
        val_values = (
            validate(model, val_loader, criterion, device, amp, amp_dtype)
            if val_loader is not None
            else None
        )
        if val_values is not None and not np.isfinite(val_values).all():
            raise FloatingPointError(f'Non-finite validation loss in epoch {epoch + 1}')
        record = {
            'epoch': epoch + 1,
            'lr': optimizer.param_groups[0]['lr'],
            'optimizer_updates': optimizer_updates,
            'skipped_updates': skipped_updates,
            'train_total': float(train_values[0]),
            'train_intensity': float(train_values[1]),
            'train_ssim': float(train_values[2]),
            'train_gradient': float(train_values[3]),
            'val_total': float(val_values[0]) if val_values is not None else None,
            'val_intensity': float(val_values[1]) if val_values is not None else None,
            'val_ssim': float(val_values[2]) if val_values is not None else None,
            'val_gradient': float(val_values[3]) if val_values is not None else None,
        }
        history.append(record)
        monitored = record['val_total'] if val_values is not None else record['train_total']
        improved = monitored < best_val
        if improved:
            best_val = monitored
        if optimizer_updates == 0:
            raise FloatingPointError(f'All optimizer updates were skipped in epoch {epoch + 1}')
        scheduler.step()
        save_checkpoint(
            output_dir / 'last.pth', model, optimizer, scheduler, scaler,
            epoch, best_val, history, args,
            [loader for loader in (train_loader, val_loader) if loader is not None]
        )
        if improved:
            save_checkpoint(
                output_dir / 'best.pth', model, optimizer, scheduler, scaler,
                epoch, best_val, history, args,
                [loader for loader in (train_loader, val_loader) if loader is not None]
            )
        with (output_dir / 'history.json').open('w', encoding='utf-8') as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
        plot_history(history, output_dir / 'loss_curves.png')
        logger.info(
            'epoch %d train=%.5f val=%s best=%.5f time=%.1fs',
            epoch + 1,
            record['train_total'],
            f"{record['val_total']:.5f}" if record['val_total'] is not None else 'N/A',
            best_val,
            time.time() - started,
        )

    logger.info('Training complete. Best checkpoint: %s', output_dir / 'best.pth')


def parse_args():
    parser = argparse.ArgumentParser(description='Train SACAFM FusionMamba')
    parser.add_argument('--ir_path', help='Infrared image directory')
    parser.add_argument('--vis_path', help='Visible image directory')
    parser.add_argument('--kaist_root', help='KAIST root containing nested lwir/visible sequences')
    parser.add_argument('--train_list', help='Text file containing training filenames')
    parser.add_argument('--val_list', help='Text file containing validation filenames')
    parser.add_argument('--output_dir', default='runs/fusion/sacafm')
    parser.add_argument('--resume', help='Path to a resumable checkpoint')
    parser.add_argument('--pretrained', help='Model checkpoint for fine-tuning; optimizer starts fresh')
    parser.add_argument('--length', type=int, default=0, help='Maximum training pairs; 0 uses all')
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='0', help='CUDA index or cpu')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False,
                        help='Opt-in mixed precision; FP32 is the reproducible default')
    parser.add_argument('--amp_dtype', choices=('bfloat16', 'float16'), default='float16',
                        help='Only applies with --amp; validate numerical stability first')
    parser.add_argument('--disable_dsdam', action='store_true', help='Ablate SACAFM alignment')
    parser.add_argument('--weighting_mode', choices=('equal', 'learned', 'acgaw'), default='acgaw',
                        help='Modality weighting ablation: fixed, unguided learned, or confidence-guided')
    parser.add_argument('--share_encoder_weights', action='store_true')
    parser.add_argument('--log_interval', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    os.makedirs('logs', exist_ok=True)
    setup_logger('logs')
    train_fusion(arguments, logging.getLogger())
