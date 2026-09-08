import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
from numpy import asarray

def imresize(arr, size, interp='bilinear', mode=None):
    numpydata = asarray(arr)
    im = Image.fromarray(numpydata, mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size) * percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size) * size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])
    return np.array(imnew)

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames

class Fusion_dataset(Dataset):
    def __init__(
        self,
        split,
        ir_path=None,
        vi_path=None,
        length=0,
        crop_size=256,
        split_file=None,
        kaist_root=None,
    ):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        self.filepath_ir = []
        self.filenames_ir = []
        self.filepath_vis = []
        self.filenames_vis = []
        self.length = length
        self.crop_size = crop_size
        
        if ir_path and vi_path:
            ir_files, _ = prepare_data_path(ir_path)
            vi_files, _ = prepare_data_path(vi_path)
            ir_by_name = {os.path.basename(path): path for path in ir_files}
            vi_by_name = {os.path.basename(path): path for path in vi_files}
            matched = sorted(set(ir_by_name) & set(vi_by_name))
            if split_file:
                with open(split_file, 'r', encoding='utf-8') as handle:
                    selected = {
                        line.strip().replace('\\', '/')
                        for line in handle
                        if line.strip() and not line.lstrip().startswith('#')
                    }
                selected_names = {os.path.basename(name) for name in selected}
                missing = sorted(selected - set(matched) - selected_names)
                if missing:
                    sample = ', '.join(missing[:5])
                    raise ValueError(f"{len(missing)} split entries have no IR/VIS pair, e.g. {sample}")
                matched = [name for name in matched if name in selected or name in selected_names]
            if not matched:
                raise ValueError(f"No filename-matched IR/VIS pairs in {ir_path} and {vi_path}")
            self.filepath_ir = [ir_by_name[name] for name in matched]
            self.filepath_vis = [vi_by_name[name] for name in matched]
            self.filenames_ir = matched
            self.filenames_vis = matched
        elif kaist_root and split in ('train', 'val'):
            root = os.path.abspath(kaist_root)
            for lwir_path in glob.glob(os.path.join(root, '**', 'lwir', '*'), recursive=True):
                if not os.path.isfile(lwir_path) or os.path.splitext(lwir_path)[1].lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    continue
                vis_path = os.path.join(os.path.dirname(os.path.dirname(lwir_path)), 'visible', os.path.basename(lwir_path))
                if not os.path.isfile(vis_path):
                    continue
                key = os.path.relpath(lwir_path, root).replace('\\', '/')
                self.filepath_ir.append(lwir_path)
                self.filepath_vis.append(vis_path)
                self.filenames_ir.append(key)
                self.filenames_vis.append(key)
            if not self.filepath_ir:
                raise ValueError(f'No KAIST lwir/visible pairs found under {kaist_root}')
            if split_file:
                with open(split_file, 'r', encoding='utf-8') as handle:
                    selected = {
                        line.strip().replace('\\', '/')
                        for line in handle
                        if line.strip() and not line.lstrip().startswith('#')
                    }
                keep = [name in selected for name in self.filenames_ir]
                self.filepath_ir = [path for path, flag in zip(self.filepath_ir, keep) if flag]
                self.filepath_vis = [path for path, flag in zip(self.filepath_vis, keep) if flag]
                self.filenames_ir = [name for name, flag in zip(self.filenames_ir, keep) if flag]
                self.filenames_vis = [name for name, flag in zip(self.filenames_vis, keep) if flag]
                if not self.filepath_ir:
                    raise ValueError(f'No KAIST pairs selected by {split_file}')
        elif split == 'train':
            # Backward-compatible KAIST layout. Explicit paths are recommended.
            if not ir_path and not vi_path:
                data_dir_ir = '/mnt/f/A-dataset/KAIST/'
                dirs = sorted(d for d in os.listdir(data_dir_ir) if not d.startswith('.'))
                for dir0 in dirs:
                    subdirs = [d for d in os.listdir(os.path.join(data_dir_ir, dir0)) if not d.startswith('.')]
                    for dir1 in subdirs:
                        req_path = os.path.join(data_dir_ir, dir0, dir1, 'lwir')
                        for file in os.listdir(req_path):
                            if file.startswith('.'):
                                continue
                            filepath_ir_ = os.path.join(req_path, file)
                            self.filepath_ir.append(filepath_ir_)
                            self.filenames_ir.append(file)
                            filepath_vis_ = filepath_ir_.replace('lwir', 'visible')
                            self.filepath_vis.append(filepath_vis_)
                            self.filenames_vis.append(file)
        else:
            raise ValueError('ir_path and vi_path are required for validation and testing')

        self.split = split
        available = len(self.filepath_ir)
        self.length = min(length, available) if length > 0 else available

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]
        image_vis_color = cv2.imread(vis_path, cv2.IMREAD_COLOR)
        if image_vis_color is None:
            raise ValueError(f"Failed to load image at {vis_path}")
        image_vis = cv2.cvtColor(image_vis_color, cv2.COLOR_BGR2GRAY)
        image_ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if image_ir is None:
            raise ValueError(f"Failed to load image at {ir_path}")
        if image_ir.shape != image_vis.shape:
            raise ValueError(
                f"Unaligned pair: IR {image_ir.shape} at {ir_path}, VIS {image_vis.shape} at {vis_path}"
            )

        if self.split == 'train':
            # Preserve object scale: crop the same region from both modalities
            # instead of squeezing the complete frame to a 256x256 square.
            crop = self.crop_size
            height, width = image_ir.shape
            if height < crop or width < crop:
                scale = max(crop / height, crop / width)
                target = (int(round(height * scale)), int(round(width * scale)))
                image_ir, image_vis = self.resize(image_ir, image_vis, target, target)
                height, width = image_ir.shape
            top = np.random.randint(0, height - crop + 1)
            left = np.random.randint(0, width - crop + 1)
            image_ir = image_ir[top:top + crop, left:left + crop]
            image_vis = image_vis[top:top + crop, left:left + crop]
            if np.random.rand() < 0.5:
                image_ir = np.fliplr(image_ir).copy()
                image_vis = np.fliplr(image_vis).copy()


        image_vis = np.ascontiguousarray(image_vis, dtype=np.float32) / 255.0
        image_ir = np.ascontiguousarray(image_ir, dtype=np.float32) / 255.0
        return (
            torch.from_numpy(image_vis).unsqueeze(0),
            torch.from_numpy(image_ir).unsqueeze(0),
        )

    def __len__(self):
        return self.length

    def resize(self, data, data2, crop_size_img, crop_size_label):
        data = imresize(data, crop_size_img, interp='bicubic')
        data2 = imresize(data2, crop_size_label, interp='bicubic')
        return data, data2
