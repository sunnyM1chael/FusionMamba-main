"""Preserve published test pools and create explicit, reviewable development splits.

MSRS uses illumination-stratified sampling within its published training pool;
this is explicitly NOT a scene-disjoint protocol. LLVIP uses filename-prefix
groups conservatively, but requires review of their scene meaning before release.
No old manifests or images are changed. All output directories must be new.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_msrs(names, seed):
    strata = defaultdict(list)
    for name in sorted(names):
        strata[Path(name).stem[-1]].append(name)
    rng = random.Random(seed)
    train, val = [], []
    for key in sorted(strata):
        values = strata[key][:]
        rng.shuffle(values)
        n = max(1, round(len(values)*0.1))
        val.extend(values[:n])
        train.extend(values[n:])
    return sorted(train), sorted(val)


def split_llvip(names, seed):
    groups = defaultdict(list)
    for name in sorted(names):
        stem = Path(name).stem
        if len(stem) != 6 or not stem.isdigit():
            raise ValueError(f'Unexpected LLVIP filename: {name}')
        groups[stem[:2]].append(name)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    # Use a deterministic prefix-group subset nearest the 10% target.
    # Dynamic programming uses group sizes only, never image labels/test outcomes.
    reachable = {0: ()}
    for key in keys:
        for total, subset in list(reachable.items()):
            reachable.setdefault(total+len(groups[key]), subset+(key,))
    total = min((n for n in reachable if 0 < n < len(names)), key=lambda n: (abs(n-len(names)*.1), n))
    selected = set(reachable[total])
    train = [n for k in keys if k not in selected for n in groups[k]]
    val = [n for k in keys if k in selected for n in groups[k]]
    return sorted(train), sorted(val), sorted(selected)


def write_split(folder, name, values):
    path = folder/(name+'.txt')
    path.write_text('\n'.join(values)+'\n', encoding='utf-8')
    return {'count':len(values), 'sha256':digest(path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/root/autodl-fs/datasets'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status':'development_only_pending_audit', 'seed':args.seed,'datasets':{}}
    for dataset in ('MSRS','LLVIP'):
        root = args.root/dataset
        train_ir = root/'train'/'ir' if dataset == 'MSRS' else root/'infrared'/'train'
        train_vis = root/'train'/'vi' if dataset == 'MSRS' else root/'visible'/'train'
        test_ir = root/'test'/'ir' if dataset == 'MSRS' else root/'infrared'/'test'
        test_vis = root/'test'/'vi' if dataset == 'MSRS' else root/'visible'/'test'
        ext = '.png' if dataset == 'MSRS' else '.jpg'
        train_names = sorted(x.name for x in train_ir.glob('*'+ext))
        test_names = sorted(x.name for x in test_ir.glob('*'+ext))
        expected = (1083,361) if dataset == 'MSRS' else (12025,3463)
        if (len(train_names),len(test_names)) != expected:
            raise ValueError(f'{dataset}: unexpected source counts')
        for ir,vis in ((train_ir,train_vis),(test_ir,test_vis)):
            if {x.name for x in ir.glob('*'+ext)} != {x.name for x in vis.glob('*'+ext)}:
                raise ValueError(f'{dataset}: unpaired source pool')
        if set(train_names)&set(test_names):
            raise ValueError(f'{dataset}: overlapping source train/test names')
        if dataset == 'MSRS':
            train,val = split_msrs(train_names,args.seed)
            method = 'published_train_only; illumination-stratified image split; NOT scene-disjoint'
            groups = None
        else:
            train,val,groups = split_llvip(train_names,args.seed)
            method = 'published_train_only; 2-digit filename prefix groups; scene semantics UNVERIFIED'
        folder = args.output/dataset.lower()
        folder.mkdir()
        meta = {name:write_split(folder,name,values) for name,values in [('train',train),('val',val),('test',test_names)]}
        meta.update({'method':method,'val_prefix_groups':groups,'train_ir_root':str(train_ir),
                     'train_vis_root':str(train_vis),'test_ir_root':str(test_ir),'test_vis_root':str(test_vis)})
        # Smoke subsets stay entirely inside development pools.
        write_split(folder,'smoke_train',train[:16])
        write_split(folder,'smoke_val',val[:4])
        report['datasets'][dataset] = meta
    report['release_gates'] = ['complete decode/hash/XML audit and resolve material annotation issues',
                               'review exact/near duplicate and scene leakage candidates',
                               'verify LLVIP prefix semantics and annotation release',
                               'record that MSRS validation split is image-level; do not claim scene independence',
                               'M3FD and KAIST require separate verified benchmark protocols']
    (args.output/'protocol.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
