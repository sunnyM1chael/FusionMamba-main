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
    def __init__(self, split, ir_path=None, vi_path=None, length=0, crop_size=256):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        self.filepath_ir = []
        self.filenames_ir = []
        self.filepath_vis = []
        self.filenames_vis = []
        self.length = length
        self.crop_size = crop_size
        
        if split == 'train':
            if ir_path and vi_path:
                ir_files, _ = prepare_data_path(ir_path)
                vi_files, _ = prepare_data_path(vi_path)
                ir_by_name = {os.path.basename(path): path for path in ir_files}
                vi_by_name = {os.path.basename(path): path for path in vi_files}
                matched = sorted(set(ir_by_name) & set(vi_by_name))
                if not matched:
                    raise ValueError(f"No filename-matched IR/VIS pairs in {ir_path} and {vi_path}")
                self.filepath_ir = [ir_by_name[name] for name in matched]
                self.filepath_vis = [vi_by_name[name] for name in matched]
                self.filenames_ir = matched
                self.filenames_vis = matched
            else:
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
            self.split = split
            available = len(self.filepath_ir)
            self.length = min(length, available) if length > 0 else available
        elif split == 'test':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split

    def __getitem__(self, index):
        if self.split == 'train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]

            image_vis = cv2.imread(vis_path)
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2GRAY)
            # if image_vis is None:
            #     raise ValueError(f"Failed to load image at {vis_path}")
            # image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2GRAY)

            image_ir = cv2.imread(ir_path,0)
            if image_ir is None:
                raise ValueError(f"Failed to load image at {ir_path}")

            if image_ir.shape != image_vis.shape:
                raise ValueError(
                    f"Unaligned pair: IR {image_ir.shape} at {ir_path}, VIS {image_vis.shape} at {vis_path}"
                )

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


            image_vis = np.asarray(Image.fromarray(image_vis), dtype=np.float32) / 255.0
            image_vis = np.expand_dims(image_vis, axis=0)

            image_ir = np.asarray(Image.fromarray(image_ir), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)

            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
            )
        elif self.split == 'test':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis = cv2.imread(vis_path)
            gray_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2GRAY)
            if image_vis is None:
                raise ValueError(f"Failed to load image at {vis_path}")

            image_ir = cv2.imread(ir_path, 0)
            if image_ir is None:
                raise ValueError(f"Failed to load image at {ir_path}")

            # image_vis = np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1)) / 255.0
            image_vis = np.asarray(Image.fromarray(image_vis), dtype=np.float32) / 255.0
            image_vis = np.expand_dims(image_vis, axis=0)
            image_ir = np.asarray(Image.fromarray(image_ir), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)

            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
            )

    def __len__(self):
        return self.length

    def resize(self, data, data2, crop_size_img, crop_size_label):
        data = imresize(data, crop_size_img, interp='bicubic')
        data2 = imresize(data2, crop_size_label, interp='bicubic')
        return data, data2
