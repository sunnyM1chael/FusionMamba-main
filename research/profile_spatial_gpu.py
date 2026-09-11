"""Inference-only GPU profiling; partial profiler FLOPs are NOT total FLOPs."""
import json
import statistics
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_fused_dataset import load_model

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.cuda.is_available()
base = Path('/root/autodl-fs/research_protocol/v1/runs/spatial_smoke_corrected_v1')
destination = base / 'gpu_profile_v1.json'
if destination.exists():
    raise FileExistsError(destination)
results = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
           'dtype': 'FP32', 'batch': 1, 'input': 'two grayscale tensors 256x256',
           'warmup': 5, 'repetitions': 20, 'tf32': False,
           'scope': 'fusion only, no IO or detector; short-run weights; inference not training', 'modes': {}}
for mode in ('none', 'independent', 'joint'):
    model = load_model(base / mode / 'best.pth', torch.device('cuda'), False, False)
    torch.manual_seed(123)
    a = torch.rand(1, 1, 256, 256, device='cuda')
    b = torch.rand_like(a)
    with torch.inference_mode():
        for _ in range(5):
            output = model(a, b)
        torch.cuda.synchronize()
        del output
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(20):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(a, b)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
            assert torch.isfinite(output).all()
            del output
        row = {'parameters_total': sum(p.numel() for p in model.parameters()),
               'mean_ms': statistics.mean(samples), 'median_ms': statistics.median(samples),
               'min_ms': min(samples), 'max_ms': max(samples),
               'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
               'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
               'samples_ms': samples}
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA],
                                        with_flops=True) as prof:
                output = model(a, b)
                torch.cuda.synchronize()
            events = prof.key_averages()
            row['partial_counted_flops'] = sum(e.flops or 0 for e in events)
            row['counted_ops'] = {e.key: e.flops for e in events if e.flops}
            row['uncounted_relevant_ops'] = [e.key for e in events if not e.flops and
                                            any(s in e.key.lower() for s in ('scan', 'deform', 'softmax', 'norm'))]
            row['flops_status'] = 'partial only; custom selective scans, deformation and other operations may be omitted'
            del output
        except Exception as exc:
            row['flops_status'] = 'unavailable: ' + str(exc)
    results['modes'][mode] = row
    print(mode, json.dumps({k: v for k, v in row.items() if k not in ('samples_ms', 'counted_ops')}), flush=True)
    del model, a, b
    torch.cuda.empty_cache()
destination.write_text(json.dumps(results, indent=2))
print('GPU_WORK_COMPLETE', str(destination), flush=True)
