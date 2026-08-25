import cv2
import numpy as np
import os
import torch
import time
from PIL import Image, ImageOps
from models.vmamba_Fusion_efficross import VSSM_Fusion as net
# from loss import Fusionloss

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ======================================================================
# ========== 【快速验证配置 - 请根据服务器路径修改这里】====================
# ======================================================================
# 1. 模型权重路径（FusionMamba-DSDAM 已训练好的权重）
MODEL_PATH = './model_last/my_cross/fusion_model.pth'

# 2. M3FD 测试图像路径
IR_DIR  = '/root/autodl-fs/datasets/M3FD_Fusion/Ir'    # 红外图像
VIS_DIR = '/root/autodl-fs/datasets/M3FD_Fusion/Vis'   # 可见光图像

# 3. 融合图像输出目录
OUTPUT_DIR = '/root/autodl-fs/datasets/M3FD_test_fusion'

# 4. 最多测试多少对（先跑5~10对验证即可）
MAX_PAIRS = 10

# 5. 支持的图片格式
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
# ======================================================================

# ========== 模型开关：必须与训练时一致 ==========
# use_dsdam: 是否使用DSDAM模块（False=不使用，True=使用）
# share_encoder_weights: 双流编码器权重共享开关
#   False(默认): IR/VIS 两套独立 encoder 权重
#   True: IR/VIS 共享同一套 encoder 权重
use_dsdam = True
share_encoder_weights = False

model = net(in_chans=1, use_dsdam=use_dsdam, share_encoder_weights=share_encoder_weights)
print(f"Model: use_dsdam={use_dsdam}, DSDAM位置=每层DFFM输入前, "
      f"share_encoder_weights={share_encoder_weights}")
print(f"Device: {device}  {'(CUDA可用)' if torch.cuda.is_available() else '(自动fallback到CPU)'}")

# 加载权重
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
print(f"✅ 权重加载成功: {MODEL_PATH}")


def imresize(arr, size, interp='bilinear', mode=None):
    numpydata = np.asarray(arr)
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


def resize(image1, image2, crop_size_img, crop_size_label):
    image1 = imresize(image1, crop_size_img, interp='bicubic')
    image2 = imresize(image2, crop_size_label, interp='bicubic')
    return image1, image2


def get_image_files(input_folder):
    valid_extensions = (".bmp", ".tif", ".jpg", ".jpeg", ".png")
    return sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)])


def fusion(input_folder_ir, input_folder_vis, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tic = time.time()
    # criteria_fusion = Fusionloss()

    ir_images = get_image_files(input_folder_ir)
    vis_images = get_image_files(input_folder_vis)

    for ir_image, vis_image in zip(ir_images, vis_images):
        path1 = os.path.join(input_folder_ir, ir_image)
        path2 = os.path.join(input_folder_vis, vis_image)

        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        img1, img2 = resize(img1, img2, [256, 256], [256, 256]) #

        img1 = np.asarray(img1, dtype=np.float32) / 255.0
        img2 = np.asarray(img2, dtype=np.float32) / 255.0

        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)

        img1_tensor = torch.from_numpy(img1).unsqueeze(0).to(device)
        img2_tensor = torch.from_numpy(img2).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            out = model(img1_tensor, img2_tensor)
            # out = torch.clamp(out, 0, 1)  
            ones = torch.ones_like(out)
            zeros = torch.zeros_like(out)
            out = torch.where(out > ones, ones, out)
            out = torch.where(out < zeros, zeros, out)
            
            
            out_np = out.cpu().numpy()
            
            out_np = (out_np - np.min(out_np)) / (np.max(out_np) - np.min(out_np))

        d = np.squeeze(out_np)
        result = (d * 255).astype(np.uint8)

        output_filename = os.path.splitext(ir_image)[0] + os.path.splitext(ir_image)[1]
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, result)

    toc = time.time()
    print('Processing time: {}'.format(toc - tic))


def fusion_fast_test():
    """
    M3FD 快速验证函数：
    - 按文件名严格匹配 Ir/Vis（不是 os.listdir 顺序 zip）
    - 最多处理 MAX_PAIRS 对
    - 复用原有 fusion() 中的预处理 / 推理 / 后处理逻辑
    - 详细打印每对处理进度和保存路径
    - 结束后统计匹配数、成功数、失败数
    """
    # ============== 1. 创建输出目录 ==============
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============== 2. 读取并按文件名匹配 ==============
    def _scan_images(folder):
        files = {}
        if not os.path.isdir(folder):
            print(f"❌ 目录不存在: {folder}")
            return files
        for fname in os.listdir(folder):
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTENSIONS:
                files[fname] = os.path.join(folder, fname)
        return files

    ir_dict = _scan_images(IR_DIR)
    vis_dict = _scan_images(VIS_DIR)

    # 按文件名取交集（严格匹配）
    matched_names = sorted(set(ir_dict.keys()) & set(vis_dict.keys()))
    total_matched = len(matched_names)

    # 截断到 MAX_PAIRS
    if total_matched == 0:
        print("❌ 没有找到任何匹配的 Ir/Vis 图像对！")
        print(f"   Ir 目录: {IR_DIR}  -> {len(ir_dict)} 张图片")
        print(f"   Vis 目录: {VIS_DIR} -> {len(vis_dict)} 张图片")
        if ir_dict:
            print(f"   Ir 示例文件名: {list(ir_dict.keys())[:3]}")
        if vis_dict:
            print(f"   Vis 示例文件名: {list(vis_dict.keys())[:3]}")
        return

    process_names = matched_names[:MAX_PAIRS]
    n_process = len(process_names)

    print(f"\n========================================================")
    print(f"📁 Ir 目录:      {IR_DIR}  ({len(ir_dict)} 张)")
    print(f"📁 Vis 目录:     {VIS_DIR} ({len(vis_dict)} 张)")
    print(f"✅ 匹配成功:     {total_matched} 对")
    print(f"🔢 本次处理:     {n_process} 对 (MAX_PAIRS={MAX_PAIRS})")
    print(f"📤 输出目录:     {OUTPUT_DIR}")
    print(f"========================================================\n")

    # ============== 3. 逐对推理 ==============
    success_count = 0
    fail_count = 0
    tic = time.time()

    for idx, fname in enumerate(process_names, start=1):
        print(f"[{idx}/{n_process}] Ir: {fname} | Vis: {fname}")

        try:
            ir_path = ir_dict[fname]
            vis_path = vis_dict[fname]

            # ---- 复用原有 fusion() 中的预处理逻辑 ----
            img1 = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
            if img1 is None:
                raise FileNotFoundError(f"无法读取 IR 图像: {ir_path}")
            if img2 is None:
                raise FileNotFoundError(f"无法读取 Vis 图像: {vis_path}")

            img1, img2 = resize(img1, img2, [256, 256], [256, 256])

            img1 = np.asarray(img1, dtype=np.float32) / 255.0
            img2 = np.asarray(img2, dtype=np.float32) / 255.0

            img1 = np.expand_dims(img1, axis=0)
            img2 = np.expand_dims(img2, axis=0)

            img1_tensor = torch.from_numpy(img1).unsqueeze(0).to(device)
            img2_tensor = torch.from_numpy(img2).unsqueeze(0).to(device)

            # ---- 推理（无梯度） ----
            model.eval()
            with torch.no_grad():
                out = model(img1_tensor, img2_tensor)
                # 与原有 fusion() 完全一致的后处理
                ones = torch.ones_like(out)
                zeros = torch.zeros_like(out)
                out = torch.where(out > ones, ones, out)
                out = torch.where(out < zeros, zeros, out)

                out_np = out.cpu().numpy()
                out_np = (out_np - np.min(out_np)) / (np.max(out_np) - np.min(out_np))

            d = np.squeeze(out_np)
            result = (d * 255).astype(np.uint8)

            # ---- 保存（文件名与输入保持一致）----
            output_path = os.path.join(OUTPUT_DIR, fname)
            cv2.imwrite(output_path, result)

            print(f"    Saved:\n     {output_path}")
            success_count += 1

        except Exception as e:
            print(f"    ❌ 失败: {fname} -> {e}")
            fail_count += 1

        print()

    # ============== 4. 统计输出 ==============
    toc = time.time()
    elapsed = toc - tic
    print("========================================================")
    print(f"📊 测试统计:")
    print(f"  ✅ 文件名匹配数量: {total_matched} 对")
    print(f"  🎯 本次处理数量:   {n_process} 对")
    print(f"  ✅ 成功推理数量:   {success_count} 张")
    print(f"  ❌ 失败数量:       {fail_count} 张")
    print(f"  ⏱️  总耗时:         {elapsed:.2f} 秒")
    if success_count > 0:
        print(f"  ⚡ 平均每张耗时:   {elapsed / success_count:.3f} 秒")
    print(f"  📁 输出目录:       {OUTPUT_DIR}")
    print("========================================================")


# ============== 保留原有完整 fusion() 调用入口注释 ==============
# if __name__ == '__main__':
#     input_folder_1 = '/mnt/f/A-dataset/KAIST/set00/V000/lwir'
#     input_folder_2 = '/mnt/f/A-dataset/KAIST/set00/V000/visible'
#     output_folder = './outputs'
#
#     fusion(input_folder_2, input_folder_1, output_folder)

# if __name__ == '__main__':
#     # 【原始 TNO 测试入口】
#     input_folder_1 = "/mnt/e/EdgeDownload/Image-Fusion-main-metrics/General Evaluation Metric/Image/Source-Image/TNO/ir"
#     input_folder_2 = "/mnt/e/EdgeDownload/Image-Fusion-main-metrics/General Evaluation Metric/Image/Source-Image/TNO/vi"
#     output_folder = "./results/TNO03"
#     fusion(input_folder_1, input_folder_2, output_folder)


if __name__ == '__main__':
    # 【M3FD 快速验证入口】—— 仅跑前 MAX_PAIRS 对，验证跨数据集推理是否正常
    fusion_fast_test()
