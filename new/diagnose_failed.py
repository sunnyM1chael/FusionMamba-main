"""
诊断 M3FD 融合失败的 274 张图片原因。
在服务器上运行: python diagnose_failed.py

排查逻辑:
  1. 找出 IR/Vis 文件名匹配的对 (与 test.py 一致)
  2. 找出输出目录中实际生成的文件
  3. 失败 = 匹配对 - 已输出
  4. 对每个失败对, 检查 IR/Vis 输入: 是否存在 / cv2 能否读取 / 尺寸是否一致 / 通道数
  5. 按原因分类统计, 写出报告 (带时间戳, 避免覆盖)
"""
import os
import sys
import cv2
from pathlib import Path
from datetime import datetime

# ===== 路径配置 (与 test.py 保持一致) =====
IR_DIR = '/root/autodl-fs/datasets/M3FD_Detection/Ir'
VIS_DIR = '/root/autodl-fs/datasets/M3FD_Detection/Vis'
OUT_DIR = '/root/autodl-fs/datasets/M3FD_Detection_fused'
EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
REPORT_DIR = '/root/autodl-tmp/fusionmamba01'  # 报告输出目录


def list_by_stem(d):
    """返回 {stem: full_path}, 按文件名 stem 索引 (跨扩展名匹配)."""
    res = {}
    if not os.path.isdir(d):
        print(f'[WARN] 目录不存在: {d}')
        return res
    for f in os.listdir(d):
        if f.startswith('.'):
            continue
        p = Path(f)
        if p.suffix.lower() in EXTS:
            # 同名多扩展名只保留第一个 (与 test.py 配对逻辑一致)
            if p.stem not in res:
                res[p.stem] = os.path.join(d, f)
    return res


def main():
    ir = list_by_stem(IR_DIR)
    vis = list_by_stem(VIS_DIR)
    out = list_by_stem(OUT_DIR)

    print('=' * 60)
    print(f'IR 文件数:   {len(ir)}')
    print(f'Vis 文件数:  {len(vis)}')
    print(f'输出文件数:  {len(out)}')

    matched = set(ir) & set(vis)
    print(f'文件名匹配对数: {len(matched)}')

    ir_only = sorted(set(ir) - set(vis))
    vis_only = sorted(set(vis) - set(ir))
    if ir_only:
        print(f'仅 IR 有 (Vis 缺失): {len(ir_only)}, 前5: {ir_only[:5]}')
    if vis_only:
        print(f'仅 Vis 有 (IR 缺失): {len(vis_only)}, 前5: {vis_only[:5]}')

    failed = sorted(matched - set(out))
    print(f'\n失败数量 (匹配但无输出): {len(failed)}')
    print('=' * 60)

    if not failed:
        print('无失败样本, 退出。')
        return

    # 诊断每个失败对
    reasons = {
        'ir_missing': [],        # IR 文件不存在
        'vis_missing': [],      # Vis 文件不存在
        'ir_read_fail': [],     # IR cv2 读取失败
        'vis_read_fail': [],    # Vis cv2 读取失败
        'shape_mismatch': [],   # IR/Vis 尺寸不一致
        'channel_abnormal': [], # 通道数异常 (IR 应 1/3, Vis 应 3)
        'read_ok': [],          # 输入完全正常 -> 失败应是推理时随机/OOM
    }

    for idx, stem in enumerate(failed):
        ip = ir.get(stem)
        vp = vis.get(stem)
        if not ip:
            reasons['ir_missing'].append(stem)
            continue
        if not vp:
            reasons['vis_missing'].append(stem)
            continue

        img_i = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
        if img_i is None:
            reasons['ir_read_fail'].append((stem, ip))
            continue
        img_v = cv2.imread(vp, cv2.IMREAD_UNCHANGED)
        if img_v is None:
            reasons['vis_read_fail'].append((stem, vp))
            continue

        if img_i.shape[:2] != img_v.shape[:2]:
            reasons['shape_mismatch'].append((stem, img_i.shape, img_v.shape))
            continue

        # 通道检查
        ci = 1 if img_i.ndim == 2 else img_i.shape[2]
        cv = 1 if img_v.ndim == 2 else img_v.shape[2]
        if ci not in (1, 3) or cv not in (1, 3):
            reasons['channel_abnormal'].append((stem, ci, cv))
            continue

        reasons['read_ok'].append((stem, img_i.shape, ci, cv))

    print('\n=== 失败原因分类 ===')
    for k, v in reasons.items():
        print(f'  {k:18s}: {len(v)}')

    # 细节打印 (每类最多 10 条)
    print('\n=== 细节 (每类前 10 条) ===')
    for k, v in reasons.items():
        if not v:
            continue
        print(f'\n-- {k} ({len(v)}) --')
        for item in v[:10]:
            print(f'  {item}')
        if len(v) > 10:
            print(f'  ... 还有 {len(v) - 10} 条')

    # 写报告
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(REPORT_DIR, exist_ok=True)
    report = os.path.join(REPORT_DIR, f'diagnose_failed_{ts}.txt')
    with open(report, 'w', encoding='utf-8') as f:
        f.write(f'IR: {IR_DIR} ({len(ir)})\n')
        f.write(f'Vis: {VIS_DIR} ({len(vis)})\n')
        f.write(f'Out: {OUT_DIR} ({len(out)})\n')
        f.write(f'匹配对: {len(matched)}, 失败: {len(failed)}\n')
        f.write('=' * 60 + '\n\n')
        for k, v in reasons.items():
            f.write(f'== {k} ({len(v)}) ==\n')
            for item in v:
                f.write(f'{item}\n')
            f.write('\n')

    print(f'\n报告已写入: {report}')
    print('=' * 60)

    # 结论提示
    n_ok = len(reasons['read_ok'])
    print('\n[结论提示]')
    if n_ok == len(failed):
        print(f'  所有 {n_ok} 张失败的输入文件均正常 -> 失败应是推理时随机/OOM,')
        print('  建议: 直接重跑这 274 张, 或用现有 3926 张训练 (足够)。')
    elif n_ok == 0 and (reasons['shape_mismatch'] or reasons['ir_read_fail'] or reasons['vis_read_fail']):
        print('  失败全部由输入文件问题导致 -> 剔除或修复这些样本即可。')
    else:
        print(f'  混合原因: {n_ok} 张输入正常 (可重跑), 其余需剔除/修复。')


if __name__ == '__main__':
    main()
