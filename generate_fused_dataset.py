"""Generate filename-aligned color fusion images for downstream YOLO training."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from split_manifest import read_manifest, paired_paths


EXTENSIONS = {'.bmp', '.tif', '.tiff', '.jpg', '.jpeg', '.png'}


def scan(folder):
    return {p.name: p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS}


def load_model(checkpoint_path, device, disable_dsdam, share_encoder_weights):
    from models.vmamba_Fusion_efficross import VSSM_Fusion

    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get('args', {}) if isinstance(checkpoint, dict) else {}
    use_dsdam = not saved_args.get('disable_dsdam', disable_dsdam)
    shared = saved_args.get('share_encoder_weights', share_encoder_weights)
    weighting_mode = saved_args.get('weighting_mode', 'acgaw')
    model = VSSM_Fusion(use_dsdam=use_dsdam, share_encoder_weights=shared,
                        weighting_mode=weighting_mode).to(device)
    state = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def infer_full(model, ir, vis, device, amp):
    ir_tensor = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0).to(device)
    vis_tensor = torch.from_numpy(vis).unsqueeze(0).unsqueeze(0).to(device)
    height, width = ir.shape
    pad_h, pad_w = (-height) % 32, (-width) % 32
    if pad_h or pad_w:
        ir_tensor = F.pad(ir_tensor, (0, pad_w, 0, pad_h), mode='replicate')
        vis_tensor = F.pad(vis_tensor, (0, pad_w, 0, pad_h), mode='replicate')
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        result = model(ir_tensor, vis_tensor)
    return result[0, 0, :height, :width].float().cpu().numpy()


def infer_tiled(model, ir, vis, device, amp, tile, overlap):
    height, width = ir.shape
    if tile <= 0 or (height <= tile and width <= tile):
        return infer_full(model, ir, vis, device, amp)
    stride = tile - overlap
    if stride <= 0:
        raise ValueError('--overlap must be smaller than --tile')
    output = np.zeros((height, width), np.float32)
    weight = np.zeros((height, width), np.float32)
    ys = list(range(0, max(height - tile, 0) + 1, stride))
    xs = list(range(0, max(width - tile, 0) + 1, stride))
    if not ys or ys[-1] != max(height - tile, 0):
        ys.append(max(height - tile, 0))
    if not xs or xs[-1] != max(width - tile, 0):
        xs.append(max(width - tile, 0))
    for top in ys:
        for left in xs:
            bottom, right = min(top + tile, height), min(left + tile, width)
            patch = infer_full(model, ir[top:bottom, left:right], vis[top:bottom, left:right], device, amp)
            output[top:bottom, left:right] += patch
            weight[top:bottom, left:right] += 1.0
    return output / np.maximum(weight, 1e-6)


def selected_names(ir_path, vis_path, split_file):
    ir_files, vis_files = scan(ir_path), scan(vis_path)
    names = sorted(set(ir_files) & set(vis_files))
    if split_file:
        names = read_manifest(split_file)
        pairs = paired_paths(ir_path, vis_path, names)
        ir_files = {name: pair[0] for name, pair in zip(names, pairs)}
        vis_files = {name: pair[1] for name, pair in zip(names, pairs)}
    elif not names or set(ir_files) != set(vis_files):
        raise ValueError('IR/VIS samples must match exactly')
    return ir_files, vis_files, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--ir_path', required=True)
    parser.add_argument('--vis_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split_file')
    parser.add_argument('--device', default='0')
    parser.add_argument('--tile', type=int, default=512, help='0 runs full-resolution inference')
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--max_pairs', type=int, default=0)
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False,
                        help='Use FP16 inference on CUDA; FP32 is the validated default')
    parser.add_argument('--disable_dsdam', action='store_true')
    parser.add_argument('--share_encoder_weights', action='store_true')
    args = parser.parse_args()

    if args.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; restore GPU or explicitly request --device cpu')
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    if args.amp and device.type != 'cuda':
        raise ValueError('--amp requires a CUDA device')
    amp = args.amp
    model = load_model(
        args.checkpoint, device, args.disable_dsdam, args.share_encoder_weights
    )
    ir_files, vis_files, names = selected_names(args.ir_path, args.vis_path, args.split_file)
    if args.max_pairs > 0:
        names = names[:args.max_pairs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, name in enumerate(names, start=1):
        ir = cv2.imread(str(ir_files[name]), cv2.IMREAD_GRAYSCALE)
        visible_color = cv2.imread(str(vis_files[name]), cv2.IMREAD_COLOR)
        if ir is None or visible_color is None:
            raise ValueError(f'Failed to load pair {name}')
        visible = cv2.cvtColor(visible_color, cv2.COLOR_BGR2GRAY)
        if ir.shape != visible.shape:
            raise ValueError(f'Unaligned pair {name}: IR={ir.shape}, VIS={visible.shape}')
        fused = infer_tiled(
            model, ir.astype(np.float32) / 255.0, visible.astype(np.float32) / 255.0,
            device, amp, args.tile, args.overlap
        )
        if not np.isfinite(fused).all():
            raise FloatingPointError(f'Non-finite fused output for {name}')
        fused_y = (fused * 255.0).round().clip(0, 255).astype(np.uint8)
        if args.grayscale:
            result = fused_y
        else:
            ycrcb = cv2.cvtColor(visible_color, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = fused_y
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        destination = output_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), result):
            raise OSError(f'Could not save {destination}')
        if index % 50 == 0 or index == len(names):
            print(f'{index}/{len(names)}')


if __name__ == '__main__':
    main()
