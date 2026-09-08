# YOLO11 + ESSD-Head 部署包

基于 Ultralytics YOLO11 的能量引导、尺度选择型 DSDAM 小目标检测头。

## 安装

```bash
pip install -e .
```

## 推荐配置

- `ultralytics/cfg/models/11/yolo11-essd.yaml`: P2-P5 四尺度检测，P2/P3 密集采样。
- `ultralytics/cfg/models/11/yolo11-dsdam.yaml`: 保留的三尺度消融配置。

不要直接对自定义 YAML 调用普通 `load()`：插入模块会改变层编号，导致多数
预训练权重无法匹配。使用训练入口完成显式权重重映射：

```bash
python train_essd.py --data ultralytics/cfg/datasets/M3FD-Fused.yaml --device 0
```

先修改数据 YAML 中的 `path`。正式实验必须让 YOLO11n 与 ESSD-Head 使用相同
的 train/val/test 划分、分辨率、增强和随机种子；建议先以 640 分辨率筛选，
再用 960 分辨率复现最终小目标结果。
