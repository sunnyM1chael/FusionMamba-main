"""
补跑 M3FD 融合失败的样本 (断点续跑)。
- 只处理 OUTPUT_DIR 中缺失的对 (即上次失败的 274 张)
- 复用 test.py 的全局模型加载与 resize 逻辑 (import test 会触发权重加载)
- 每张推理后清理显存, 避免长时间运行导致 GPU 累积失败

运行: python resume_failed.py
"""
import os
import time

import cv2
import numpy as np
import torch

# import test 会执行 test.py 全局代码: 加载 model / device / resize / VALID_EXTENSIONS
# (test.py 的 fusion_fast_test 在 if __name__=='__main__' 中, import 时不会自动跑)
from test import model, device, resize, VALID_EXTENSIONS

# ===== 路径配置 (与服务器 M3FD_Detection 一致) =====
IR_DIR = '/root/autodl-fs/datasets/M3FD_Detection/Ir'
VIS_DIR = '/root/autodl-fs/datasets/M3FD_Detection/Vis'
OUTPUT_DIR = '/root/autodl-fs/datasets/M3FD_Detection_fused'


def scan_images(folder):
    files = {}
    if not os.path.isdir(folder):
        print(f'❌ 目录不存在: {folder}')
        return files
    for fname in os.listdir(folder):
        if os.path.splitext(fname)[1].lower() in VALID_EXTENSIONS:
            # 与 test.py 的 _scan_images 一致: 同名取第一个
            if fname not in files:
                files[fname] = os.path.join(folder, fname)
    return files


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ir_dict = scan_images(IR_DIR)
    vis_dict = scan_images(VIS_DIR)

    matched = sorted(set(ir_dict) & set(vis_dict))
    print(f'文件名匹配对数: {len(matched)}')

    # 只处理输出缺失的对 (断点续跑)
    todo = [f for f in matched if not os.path.exists(os.path.join(OUTPUT_DIR, f))]
    print(f'需补跑 (输出缺失): {len(todo)}')
    print('=' * 56)

    if not todo:
        print('✅ 全部已完成, 无需补跑')
        return

    success, fail = 0, 0
    failed_names = []
    tic = time.time()
    model.eval()

    for idx, fname in enumerate(todo, 1):
        print(f'[{idx}/{len(todo)}] {fname}')
        try:
            img1 = cv2.imread(ir_dict[fname], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(vis_dict[fname], cv2.IMREAD_GRAYSCALE)
            if img1 is None:
                raise ValueError('IR 读取失败 (None)')
            if img2 is None:
                raise ValueError('Vis 读取失败 (None)')

            # 保持原始分辨率 (不 resize), 与服务器 test.py 一致
            img1 = np.asarray(img1, dtype=np.float32) / 255.0
            img2 = np.asarray(img2, dtype=np.float32) / 255.0
            img1 = np.expand_dims(img1, 0)
            img2 = np.expand_dims(img2, 0)
            t1 = torch.from_numpy(img1).unsqueeze(0).to(device)
            t2 = torch.from_numpy(img2).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(t1, t2)
                ones = torch.ones_like(out)
                zeros = torch.zeros_like(out)
                out = torch.where(out > ones, ones, out)
                out = torch.where(out < zeros, zeros, out)
                out_np = out.cpu().numpy()
                # 归一化 (与 test.py 一致)
                out_np = (out_np - np.min(out_np)) / (np.max(out_np) - np.min(out_np))

            d = np.squeeze(out_np)
            result = (d * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(OUTPUT_DIR, fname), result)
            success += 1
            print(f'  ✅ Saved: {os.path.join(OUTPUT_DIR, fname)}')

        except Exception as e:
            print(f'  ❌ 失败: {fname} -> {e}')
            fail += 1
            failed_names.append((fname, str(e)))

        # 每张后清理显存, 防止长时间运行累积导致后续连续失败
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - tic
    print('=' * 56)
    print(f'📊 补跑统计:')
    print(f'  ✅ 成功: {success}')
    print(f'  ❌ 失败: {fail}')
    print(f'  ⏱️  耗时: {elapsed:.1f}s')
    print(f'  ⚡ 均耗: {elapsed / max(success, 1):.3f}s/张')

    if fail > 0:
        print(f'\n⚠️ 仍有 {fail} 张失败, 具体原因:')
        for fname, e in failed_names:
            print(f'   {fname}: {e}')
        print('\n建议: 若是 OOM, 重启实例后只跑这些; 若是数据问题, 剔除即可。')


if __name__ == '__main__':
    main()
