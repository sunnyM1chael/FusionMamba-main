"""Strict, portable sample manifests shared by training and evaluation."""

from pathlib import Path, PurePosixPath
import hashlib


def read_manifest(path):
    names = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        name = line.strip().replace('\\', '/')
        if not name or name.startswith('#'):
            continue
        parts = PurePosixPath(name)
        if parts.is_absolute() or '..' in parts.parts or ':' in name:
            raise ValueError(f'Unsafe relative sample path: {name}')
        names.append(parts.as_posix())
    if not names:
        raise ValueError(f'Empty manifest: {path}')
    if len(set(names)) != len(names):
        raise ValueError(f'Duplicate entries in manifest: {path}')
    return names


def manifest_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_paths(ir_root, vis_root, names):
    pairs = [(Path(ir_root) / name, Path(vis_root) / name) for name in names]
    missing = [name for name, pair in zip(names, pairs) if not all(p.is_file() for p in pair)]
    if missing:
        raise ValueError(f'{len(missing)} manifest entries lack an IR/VIS pair: {missing[:5]}')
    return pairs
