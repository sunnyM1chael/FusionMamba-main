# 科研项目长期上下文

> 本文保存长期有效的研究定义、技术路线、数据职责、实验事实与决策边界。
> 阶段性进度和下一步执行状态见 `CURRENT_STATUS.md`。本文中的“验证集结果”均为
> 开发阶段证据，除非明确写明，否则不是最终测试集结论。

## 1. 研究背景

### 1.1 学科方向与研究主题

本项目属于计算机视觉、多模态图像处理与深度学习目标检测方向，研究主题是：
红外与可见光图像先经融合网络生成单幅融合图像，再由目标检测网络完成目标识别。
研究内容作为学位论文中的一个完整章节，融合和检测属于同一条实验链，不拆成两个
彼此独立的章节。

论文目标不是单独追求融合图像视觉观感，也不是只改检测器，而是在统一数据和评价
协议下回答以下问题：

1. 融合端的空间增强/对齐设计是否保留了对下游检测有用的信息；
2. 自适应模态权重是否真正随模态可靠性变化，而不只是产生非恒定权重；
3. 检测端 DSDAM 是否能在具有足够小目标的多模态数据上提高识别能力，并与 P2
   检测头的贡献分离；
4. 完整“融合→检测”方案能否在单模态基线、内部消融和代表性外部方法对比中成立。

### 1.2 已确定的研究路线

当前路线固定为两阶段、非端到端联合优化：

`配对红外/可见光 → FusionMamba 融合前端 → 融合图像 → YOLO11s 检测器 → 检测评价`

融合模型先独立训练并冻结，再离线生成融合图像；检测器在相同数据划分、初始化和
训练设置下独立训练。融合损失没有检测监督，因此融合目标与检测目标是否一致必须
通过下游实验验证，不能由融合损失或主观图像质量替代。

目前规划的论文贡献主线是：

- 融合端：以 SADFFM 为工作论文名的空间自适应融合模块；代码兼容名为 SACAFM。
  该贡献仍是候选，已有单种子下游正向证据，但尚未完成跨种子确认。
- 检测端：DSDAM 与 P2 组成的 ESSD-Head，用 2×2 因子实验分离 DSDAM、P2 及二者
  交互作用。该贡献已完成工程预检，尚未完成 M3FD 正式消融。
- ACG-AW 自适应权重已经完成开发筛选，但没有超过普通可学习权重，当前不作为
  已成立的核心创新点，也不为凑创新数量继续强行调参。

## 2. 技术路线

### 2.1 整体 pipeline

1. 数据输入：配准或近似配准的红外图像与可见光图像。融合端主要在 MSRS 开发；
   下游行人检测在 LLVIP 开发；检测端小目标消融计划在 M3FD 开展。
2. 融合编码：双流 VSSM/Mamba 编码器分别提取红外和可见光特征。
3. 分层融合：每个编码 stage 在下采样前完成可选空间增强、模态加权和 DFFM 交互，
   生成送往解码器的多尺度融合 skip 特征。融合输出不会替代下一 stage 的原始双流
   编码输入，因此当前实现不是逐 stage 累积的几何配准网络。
4. 图像生成：解码器恢复融合亮度图；当前正式链路输出无损灰度 PNG，检测加载器
   扩展为三通道，不重新引入可见光色度。
5. 目标检测：标准 YOLO11s 或加入 DSDAM/P2 的 ESSD 变体。
6. 评价：融合质量指标作为支持证据，以 AP@[.50:.95] 为检测主指标，同时报告 AP50、
   Precision、Recall、逐类指标、小目标指标、参数量、延迟和显存。

融合训练目标为：

`L = 10 × L_SSIM + 10 × L_intensity-MSE + L_gradient-L1`

其中结构项对两种源图等权，强度目标取逐像素两模态最大值，梯度目标取逐像素最大
梯度。该目标没有目标框或检测置信度反馈。

### 2.2 融合 baseline 与改进版本

#### 受控 baseline A

- 名称：A / DFFM / no spatial enhancer。
- 作用：作为当前重构架构中的最简空间对照。
- 结构：双流编码器；每个 stage 在下采样前直接进入 DFFM；权重固定为红外/可见光
  等权；无 DSDAM 空间增强。
- 边界：A 是本项目当前代码中的受控 baseline，不得未经架构逐项核验就称为“原始
  FusionMamba 完整复现”。

#### 控制版本 B

- 名称：B / independent / self-conditioned DSDAM + DFFM。
- 目的：判断加入空间增强本身是否有用。
- 结构：同一个共享 DSDAM enhancer 分别处理两模态，每个 offset 只由本模态决定；
  位置在每个 encoder stage 后、DFFM 前；模态权重固定等权。
- 边界：并不是两套独立参数的 DSDAM；B 与 C 的 offset predictor 结构和有效范围
  不完全参数匹配。

#### 当前融合候选 C

- 论文工作名：SADFFM；代码类名：SACAFM（Shape-Adaptive Cross-modal Alignment
  and Fusion Module）。英文全称在最终论文中尚需统一，不能把两个名字写成两个模块。
- 设计目的：在位置级跨模态混合前利用联合红外、可见光、差分和乘积特征预测两组
  有界 deformable offsets，再执行 DFFM，以缓解局部对应偏差并保留任务相关信息。
- 插入位置：四个编码 stage 的单模态 VSS/VSSM block 后、DFFM 前、下采样前。
- 与 A 的区别：A 无空间增强；C 使用跨模态联合条件的空间增强。
- 与 B 的区别：B 的 offset 自条件于单模态，C 的 offset 联合条件于两模态关系。
- 当前状态：C 的干净 MSRS 融合损失不如 A，也未在合成位移上胜过 B；但在 LLVIP
  单种子下游检测上比 A 高 0.5493 个 AP50:95 百分点。因此只保留为“有下游潜力的
  候选”，尚不能声称几何对齐或普遍优越性已成立。

#### 模态加权版本

- Equal：红外/可见光固定 0.5/0.5，是最小参照。
- Learned：普通可学习 gate，不引入对齐置信度。
- ACG-AW：Alignment-Confidence-Guided Adaptive Weighting。使用特征差异形成启发式
  confidence，并缩放两模态 logits；在 DFFM 前加权，同时影响残差融合输出。
- 已证实 ACG-AW gate 会变化且会影响输出；未证实它能在模态受损时正确降低受损
  模态权重。它在旧布局权重对照中没有超过 Learned，当前不作为已验证创新。

### 2.3 检测 baseline 与改进版本

#### 检测 baseline

- 名称：YOLO11s baseline。
- 作用：标准 P3–P5 三尺度检测；作为所有检测端消融的共同基线。
- 选择原因：比 YOLO11n 更适合作为学位论文主模型，同时计算量仍可控；已用于 LLVIP
  四输入对照。YOLO11n 只保留为快速 smoke 工具。

#### DSDAM

- 名称：Deformable Spatial Dual-Attention Module。
- 设计目的：在重复下采样前增强小目标的局部空间证据。
- 结构：卷积初始化、能量引导图、offset predictor、torchvision DeformConv2d、
  windowed multi-head self-attention、残差、BatchNorm 与 ReLU。
- 插入位置：检测 backbone 的 stride 4、8、16 特征处，即 P2/P3/P4 对应层。
- 重要限制：DeformConv2d 的 CUDA backward 不支持严格确定性；必须固定 seed 并对
  核心结果做多随机种子均值/标准差。

#### P2 与 ESSD-Head

- P2-only：标准 backbone，增加 P2 检测输出，形成 P2–P5 四尺度头。
- DSDAM-only：在 stride 4/8/16 加 DSDAM，仍使用标准 P3–P5 检测头。
- ESSD：Energy-Guided Scale-Selective DSDAM Head；同时使用 DSDAM 和 P2–P5 检测头。
- 正式设计：baseline、P2-only、DSDAM-only、ESSD 构成 2×2 因子实验。
- 参数量并不相等，因此结果必须同时报告参数、FLOPs/未覆盖算子说明、显存与延迟，
  不得声称完全排除了容量因素。

YOLO11s、M3FD 六类时当前构建参数量：baseline 9,430,114；P2-only 9,576,072；
DSDAM-only 13,395,675；ESSD 13,541,633。

### 2.4 已弃用或停止扩展的方案

- 融合注意力错误 NCHW 恢复排列的旧实现：已修复；旧 checkpoint 只能作为历史
  开发证据，不能加载到新实现后冒充公平复现。
- ACG-AW 作为独立核心创新：当前停止；没有验证可靠性响应，也没有超过 Learned。
- 固定 `0.75C + 0.25IR` 残差混合作为新设计：CPU 干预结果与均值/方差对照混杂，
  stop gate 已触发，不启动完整训练。
- 仅凭融合损失选择模型：已排除；C 的融合损失更差但下游 AP 更好，必须保留下游
  检测评价。
- 只调检测阈值解释 IR 与 C 差距：已排除；六组置信度/IoU设置中 IR 匹配数均更高，
  且 AP 不是单阈值指标。
- LLVIP 上建立标准小目标结论：已排除；冻结验证集按 resized-640 area<32² 仅 13 框。
- M3FD 逐图随机划分或沿用 3141/785/274：已排除，存在场景泄漏风险且来源不成立。
- 把 M3FD_Fusion 300 当独立外部集：已排除，300/300 与 4,200 主集合像素重复。
- 用不完整 KAIST 做正式泛化结论：当前不做；完整官方协议和数据尚未准备齐。

## 3. 数据集与实验环境

### 3.1 数据集职责与状态

| 数据集 | 规模与已确认信息 | 主要用途 | 当前划分 | 是否冻结 |
|---|---|---|---|---|
| MSRS | 官方训练池 1,083 对，测试 361 对 | 融合训练、权重/空间模块开发 | 974 train / 109 internal val；361 test 保留 | 开发划分已冻结；109 为照明分层逐图划分，非场景独立，需披露 |
| LLVIP | 12,025 官方 train，3,463 官方 test；42,437 个有效前景框前另发现 5 个零宽框 | “融合→行人检测”开发、VIS/IR/A/C 四输入对照 | 官方 train 内 10,825 train / 1,200 internal val；官方 test 未用于当前模型选择 | 当前开发名单与标签哈希已冻结；prefix 语义未证实为场景 |
| M3FD Detection | 4,200 IR/VIS/XML 三元组；34,407 框；六类 | 检测端 DSDAM/P2 小目标正式消融 | 计划 `M3FD-SG-70/10/20-v1`，目标约 2,940/420/840，必须按完整 scene/group 分配 | 尚未冻结；当前最高优先级 |
| M3FD Fusion 300 | 300 对 | 不作为独立测试集 | 300/300 与主 4,200 集像素重复 | 隔离，不进入独立泛化结论 |
| TNO | 当前可用 37 对 | 仅补充外部融合评价/定性展示 | 不承担训练或主要结论 | 不作为正式主测试；历史上接触过 TNO 结果须披露 |
| KAIST | 服务器仅有部分集合 | 未来可选跨数据集检测扩展 | 旧的全序列随机划分作废 | 未冻结、非当前前置任务 |
| LLVIP/MSRS 之外下载数据 | 用户曾准备 LLVIP、MSRS、M3FD、TNO、KAIST | 只有经过协议审计的数据才能进入正式实验 | 不以文件夹存在代替协议完成 | 逐项审计 |

M3FD 标注统计：People 11,477；Car 18,296；Lamp 2,405；Bus 700；Motorcycle 521；
Truck 1,008。长边等比例缩放到 640 后，area<32² 有 20,364 框，area<16² 有 10,860
框；People 对应 8,450/5,139。该数据量适合 DSDAM 小目标研究，但这些阈值是项目预
声明代理，不是 M3FD 官方指标。

M3FD 官方论文/仓库没有提供可复现的 train/val/test 成员表；作者代码可从
`scenario.json` 生成 train/val，但官方仓库和当前数据包都没有该元数据，也没有正式
test 语义。因此自建协议必须明确称为 `M3FD-SG-70/10/20-v1`，不能称 official split。

### 3.2 环境

#### 本机

- 操作系统：Windows；常用 PowerShell。
- 工作区：`E:\EdgeDownload\FusionMamba-main`。
- M3FD 本地数据：`E:\EdgeDownload\Dataset\M3FD_Detection`。
- 用户原始 LLVIP 压缩包位置曾为：`F:\A-dataset\LLvip\LLVIP.zip`。
- 本机项目根目录的 `yolo11s.pt` 是 1,471,531 字节的截断 ZIP，禁止使用或上传为
  正式权重。

#### 服务器

- SSH：`ssh -p 42323 root@connect.cqa1.seetacloud.com`。
- 操作系统：Ubuntu 22.04.4 LTS。
- 训练硬件：有卡模式为单张 NVIDIA RTX 4090 D；当前交接时处于无卡/CPU模式。
- 当前基础环境实测：Python 3.12.3，PyTorch 2.5.1+cu124，torchvision 0.20.1+cu124，
  CUDA build 12.4；无卡模式下 `torch.cuda.is_available()` 为 False。
- 当前服务器环境可导入的 Ultralytics 源版本标识：8.4.126。历史检测计划中记录的官方 YOLO11s
  权重来自 Ultralytics v8.4.0/v8.3.0 asset，文件内容哈希一致。
- 官方 YOLO11s 权重服务器位置：
  `/root/autodl-fs/research_protocol/v1/pretrained/yolo11s.pt`。
- YOLO11s SHA-256：
  `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`。
- 权威实验代码工作副本：`/root/autodl-tmp/FusionMamba-SACAFM-14e9a93`。
- 版本化数据/结果根：`/root/autodl-fs/research_protocol/v1`。
- 服务器上还存在其他 checkout/editable 安装；运行时必须从权威目录调用脚本并核验
  实际 import 路径，避免加载 `/root/autodl-tmp/fusionmamba_git/...` 的旧代码。

README 中 Python 3.8、torch 1.13、CUDA 11.7 是上游/历史安装说明，不代表当前服务器
实测环境。后续正式实验必须保存完整环境快照，不能混用两个版本描述。

当前 ESSD 正式入口的显式默认值为：YOLO11s、img640、150 epochs、batch16、
workers8、AdamW、lr0=0.001、lrf=0.01、weight decay=0.0005、warmup=3、cosine、
close_mosaic=10、patience=0、AMP开启、cache=False、rect=False、plots=False、
pretrained=False（因为手工映射预训练张量）、seed42、strict deterministic=False。
若 smoke 表明 AMP 对 DSDAM 不稳定，应在正式训练前统一改为四组 FP32，而不是逐组
切换。融合前端由于已有 FP16/BF16 异常，保持 FP32。

### 3.3 Git 与重要文件

- Git 分支：`codex/sacafm-essd`。
- Git 远端：`https://github.com/sunnyM1chael/FusionMamba-main.git`。
- 本交接创建前最新科研提交：`9b508e3`；M3FD 全序列证据提交为 `fac9c1e`，ESSD
  公平性与 M3FD 协议提交为 `1c191fe`。
- 服务器权威工作副本不是干净的独立 Git checkout；不要在服务器根目录直接执行
  reset/clean。以本机分支提交为版本依据，向服务器同步明确文件。

关键文件：

- 融合网络：`models/vmamba_Fusion_efficross.py`。
- 融合空间模块：`DSDAM.py`。
- DFFM/ACG-AW：`models/cross.py`。
- 融合训练：`train.py`，正式启动辅助 `research/launch_msrs.py`、
  `research/spatial_formal.py`。
- 融合损失：`loss.py`。
- 融合导出：`generate_fused_dataset.py`。
- 检测端 DSDAM：`yolo11_dsdam_deploy/ultralytics/nn/modules/DSDAM.py`。
- ESSD 训练入口：`yolo11_dsdam_deploy/train_essd.py`。
- 四组 YAML：标准 `yolo11.yaml`、`yolo11-essd-p2-only.yaml`、
  `yolo11-essd-dsdam-only.yaml`、`yolo11-essd.yaml`。
- M3FD 审计：`research/audit_m3fd_small_objects.py`、`audit_m3fd_pairs.py`、
  `audit_m3fd_sequence.py`。
- 检测失败分析：`research/analyze_detection_errors.py`、
  `analyze_image_evidence.py`、`analyze_failure_mechanisms.py`、
  `check_edge_preservation.py`、`run_cpu_input_probe.py`。
- 长期实验规范：`research/PROTOCOL.md`。

## 4. 已完成实验总览

### 4.1 数据完整性审计

- 目的：防止图像缺失、配对错位、无效标注、重复数据和不合法 test 使用。
- 设置：服务器只读扫描 21,432 对样本；图像解码、尺寸、像素/dHash、标注统计。
- 结果：MSRS/LLVIP 配对解码通过；LLVIP 发现 5 个零宽框并在派生 YOLO 标签中拒绝；
  两个空标注样本保留为负样本；M3FD 4,200 三元组齐全、尺寸全部匹配。
- 异常：M3FD_Fusion 300 与主集合全部重复；M3FD 无可信划分元数据。
- 影响：所有正式实验必须使用冻结 manifest；M3FD 正式训练前必须完成场景分组。

### 4.2 融合/检测工程 smoke 与布局修复

- 目的：确认代码路径可训练、可保存、可恢复、可验证。
- 设置：融合 16 train/4 val、crop128、batch4、FP32、3 epochs；检测 LLVIP 16/4、
  YOLO11n、img320、1 epoch。
- 结果：Equal/Learned/ACG-AW 均跑通；baseline/P2-only/DSDAM-only/ESSD 均完成训练、
  best 保存、隔离重载和验证；端到端小样本 Fusion→ESSD 链路跑通。
- 异常与修复：检测和融合 DSDAM 都曾存在 attention 输出恢复 NCHW 的排列错误；
  正确形式为 `permute(0,1,3,2).contiguous().view(B,C,H,W)`。融合旧实现分析探针中
  22/24 元素不匹配；修复后 global/window1/window2 和梯度测试通过。旧结果不得与
  修复后模型混用。
- 精度异常：FP16 在融合 smoke 中跳过早期 optimizer step，BF16 出现非有限 loss；
  因此融合正式训练固定 FP32。

### 4.3 首个 MSRS 全模型开发运行（历史旧布局）

- 目的：建立 SACAFM+ACG-AW 完整模型的初始开发记录。
- 设置：974 train/109 val/361 test；RTX 4090 D；crop256；batch4；FP32；seed42；
  100 epochs；commit `4cfda49`。
- 结果：val loss 2.11808→best 1.87147（epoch74），final 1.87199；24,300 updates，
  zero skipped。361 test 融合指标：EN 6.44875、SD 38.31976、SF 10.51291、AG
  3.98132、MI 2.54361、source-average SSIM 0.71035。
- 异常：该批属于后来确认的旧 attention layout，不能代表修复后最终方案，也不能
  单独证明相对 FusionMamba 提升。
- 影响：保留为历史开发证据和完整性记录，不用于最终创新排序。

### 4.4 权重消融（历史旧布局）

- 目的：比较 Equal、普通 Learned 与 ACG-AW。
- 设置：同一 MSRS 974/109 开发集；100 epochs；seed42；crop256；batch4；FP32；
  总损失相同。
- 结果：Equal 1.872236382；Learned 1.870807475；ACG-AW 1.871465321。
  Learned 最低，ACG-AW 未超过 Learned。
- 行为诊断：109 张完整验证图上，Learned 平均 IR 权重 0.47067、偏离 0.5 的平均
  绝对值 0.09905、距0.5不足0.01比例0.23523；ACG-AW 分别0.48855、0.15426、
  0.10743。前8张替换等权后输出平均绝对变化 Learned 0.04314、ACG-AW 0.02864。
  ACG-AW 仅在 41/109 张上 loss 低于 Learned。
- 异常与恢复：历史权重实验曾修复 checkpoint 保存/恢复流程；恢复后只完成了权重可加载
  与行为检查。这也是该批结果必须按“旧 attention layout 的开发筛选”而非最终论文证据
  管理的原因之一。
- 结论：权重路径确实生效，但没有验证“正确响应模态可靠性”；ACG-AW 不作为已
  成立创新。不能把无收益归因于空间对齐差，因为没有该因果证据。

### 4.5 修复后空间模块 A/B/C 正式开发消融

- 目的：比较无空间增强、自条件 DSDAM、联合条件 SADFFM。
- 设置：MSRS 974/109；test 361 不参与；Equal weighting；seed42；FP32；crop256；
  batch4；100 epochs；公共同名同形状张量初始化一致。
- 结果：A 1.864409649（epoch93）；B 1.867945544（epoch89）；C 1.866732974
  （epoch91）。C 比 B 低 0.0649%，但比 A 高 0.1246%。C-A paired bootstrap
  +0.0023233，95% CI `[+0.0017820,+0.0028985]`；C 仅胜 A 22/109。
- 合成位移：visible 向右/向下 2/4/8 pixel、replicate padding、边缘 crop8。B 在六个
  shifted 条件中五个最好；C 未证明优于 B。匹配 crop8 的 right8/down8 增量：
  A 0.783332/0.960756；B 0.775639/0.949641；C 0.778350/0.955815。
- 成本：A/B/C 参数 320.38M/341.27M/342.17M；256² batch1 FP32 融合前向
  43.06/55.35/57.11 ms；峰值显存 1416.49/1788.16/1792.70 MiB。FLOPs 只部分
  覆盖，不能作为完整 FLOPs。
- 结论：融合 loss 和合成位移不支持 C 作为已验证的几何对齐创新；保留到下游检测
  再判断任务信息价值。

### 4.6 LLVIP A/C 下游检测

- 目的：检验融合 loss 是否遗漏了任务有用信息。
- 设置：A/C 融合 checkpoint 在 MSRS val 上预先选择；分别导出 LLVIP 融合图；
  YOLO11s、10,825 train/1,200 val、seed42、img640、batch16、FP32、150 epochs、
  相同初始化。官方 test 3,463 未使用。
- 训练记录最佳：A AP50:95 0.616212，C 0.621705；C 增加 0.005493，即 0.5493
  个百分点；Recall 增加 0.012446，AP50 略降 0.000434。
- 一致重评近似值：A 0.616271，C 0.621695。小的末位差异来自后续统一验证/导出
  记录，论文中必须选定一套 evaluator 后统一重算，不能混用小数末位。
- 异常：C 融合图导出曾在 10,055/10,825 时因数据盘约 200,000 inode 配额耗尽中断；
  清理可再生的小文件并恢复后完成。曾生成的派生 visible 视图先归档再移除，不能把这次
  基础设施中断误判为模型失败，也不能据此改变实验成员表。
- 结论：C 比 A 有单种子下游正向证据，但不能证明几何对齐，也没有跨 seed 稳定性。

### 4.7 LLVIP 单模态对照

- 目的：判断“融合→检测”是否超过最强单模态。
- 设置：与 A/C 完全相同的 YOLO11s、划分、初始化、seed42、150 epochs、FP32。
- 结果（一致重评）：VIS 0.510612；IR 0.636271；A 0.616271；C 0.621695。
  原训练记录 IR 0.636254、C 0.621705、A 0.616212；VIS best 重验约 0.510568。
- 结论：C 比 VIS 高 11.1083 个百分点左右，但比 IR 低 1.4576 个百分点左右；当前
  没有证明融合优于直接红外检测。该负结果必须保留，不能只报告 C>A。

### 4.8 融合低于 IR 的失败分析

- 目的：定位是置信度、误检、定位、对比度、边缘、补边还是目标不一致造成差距。
- 设置：LLVIP val 1,200 图/3,778 GT，四个 seed42 detector；未重训、未用 test。
- conf0.25/IoU0.5：IR TP3643/FP457/FN135；C TP3615/FP400/FN163。IR-only 80，
  C-only 52，both 3563，neither 83。
- IR-only 80 中：52 个有 IoU≥0.5 但置信度不足0.25；16 个有重叠但 IoU<0.5；
  12 个受一对一匹配竞争。28/80 接触图像边界，C-only 仅1/52。
- both 目标的最高候选 IoU 均值 IR/C 为0.8372/0.8313。
- C 相对 IR 对比度下降是普遍现象：IR-only -0.0421，C-only -0.0642；成功方向
  下降更明显，因此“对比度下降”不是独立失败原因。融合增加局部梯度/纹理，同时
  对红外最强10%梯度强度平均降低约10%，但两类保留率约93.5%，不能声称强边缘
  被普遍破坏。
- 导出 padding 不是原因：所有 val 图为1024×1280，均可被32整除，补边分支未触发。
- 结论：C 更保守、FP 较少但 FN 较多，并有轻微定位劣势；融合损失与检测目标不
  完全一致是合理假设，但未证明唯一因果或 DSDAM 对齐失败。

### 4.9 固定 IR 残差 CPU 机制探针

- 目的：在不重训情况下检查 C 中加入固定 IR 成分是否特异性恢复失败目标。
- 设置：24 个 outcome-selected 图像×3输入；原C、0.75C+0.25IR、与混合图均值/
  方差近似匹配的C；CPU one-thread；C detector 固定。
- v1 rect=True 512×640 与正式 validator 不一致，只能作路径内探索。
- v2 修正为 rect=False 640×640：IR-only hit 1/2/2，C-only 7/7/7，both 8/8/8；
  IR-only mean score 0.20321/0.30497/0.20379。
- 结论：额外 hit 同样出现在 moment control，不能证明 IR 空间残差的特异作用；
  C-only 也未改善。完整训练 stop gate 已触发。

### 4.10 M3FD 审计与 ESSD 公平性预检

- 目的：为检测端小目标正式消融建立数据和初始化条件。
- M3FD：4,200 对全部完整；34,407 框；尺寸/格式匹配；全序列 4,199 个相邻变化
  已记录；top80 变化人工查看表明既有真实场景切换也有普通大运动，不能自动切分。
  发现58个双模态 equal-dHash 候选组，必须在 split 时保持同组并做全分辨率确认。
- ESSD：YOLO11n 与 YOLO11s 四变体均转移378个共同预训练张量，source hash 在
  各变体内一致；YOLO11s source-tensor hash 前缀 `5b0296b8d606`。本机和CPU服务器
  preflight 通过。
- 工程修复：四变体统一为 single-build trainer 和隔离 best validation；固定 scale s、
  150 epochs、patience0，并显式记录训练参数、checkpoint hash、共同张量 hash、完整
  初始化模型 hash；最终验证固定 split=val、rect=False、half=False。
- 结论：工程条件已基本满足，但这不是正式 DSDAM 实验结果；M3FD split 未冻结前
  禁止启动正式训练。

## 5. 核心实验结果表

| 实验 | 指标 | 结果 | 研究结论 |
|---|---:|---:|---|
| 权重 Equal | MSRS val fusion loss | 1.872236382 | 历史旧布局参照 |
| 权重 Learned | MSRS val fusion loss | **1.870807475** | 三者最低 |
| 权重 ACG-AW | MSRS val fusion loss | 1.871465321 | 未超过 Learned，可靠性响应未证实 |
| 空间 A | MSRS val fusion loss | **1.864409649** | 修复后干净集最佳 |
| 空间 B | MSRS val fusion loss | 1.867945544 | 合成位移六项中五项最佳 |
| 空间 C/SADFFM | MSRS val fusion loss | 1.866732974 | 比B略好、比A差 |
| A 下游检测 | LLVIP val AP50:95 | 0.616271 | 一致重评值 |
| C 下游检测 | LLVIP val AP50:95 | 0.621695 | 比A高约0.549个百分点 |
| IR-only 检测 | LLVIP val AP50:95 | **0.636271** | 比C高约1.458个百分点 |
| VIS-only 检测 | LLVIP val AP50:95 | 0.510612 | C明显高于VIS |
| IR错误统计 | TP/FP/FN @.25/.5 | 3643/457/135 | 更多命中、更多FP |
| C错误统计 | TP/FP/FN @.25/.5 | 3615/400/163 | 更少FP、更多FN |
| CPU probe v2 IR-only | hits C/blend/control | 1/2/2 | 不支持IR残差特异机制 |
| A/B/C融合延迟 | ms, 256² FP32 b1 | 43.06/55.35/57.11 | C最慢，成本需报告 |
| M3FD小目标 | resized640 area<32² | 20,364 | 足够支持检测端小目标消融 |
| ESSD YOLO11s参数 | baseline/P2/D/ESSD | 9.43/9.58/13.40/13.54M | 非等参数，需效率对照 |

当前尚无正式外部方法对比结果，也没有检测端 DSDAM 的正式 AP 结果；不得在表中
填入 smoke 指标冒充论文结果。

## 6. 已确认事实与待验证事项

### 6.1 已验证

- 修复后 C/SADFFM 在单个 seed42 的 LLVIP 下游 AP50:95 比 A 高约0.549个百分点。
- C/SADFFM 在干净 MSRS fusion loss 上不如 A，在合成位移总体上不如 B。
- 当前 LLVIP 开发实验中，IR-only AP50:95 高于 C 约1.46个百分点。
- ACG-AW gate 非恒定且参与输出，但旧布局实验中没有超过普通 Learned gate。
- 当前证据不能证明 ACG-AW 正确响应模态可靠性。
- LLVIP 冻结验证集不足以支撑标准小目标结论；M3FD 小目标数量充足。
- M3FD 当前无可复现官方三分成员表，逐图随机划分不合格。
- DSDAM×P2 四结构能够构建、训练、保存、重载；YOLO11s共同预训练张量初始化已核验。
- 融合 loss 与检测 AP 排序不一致，因此论文必须同时保留融合质量与检测评价。

### 6.2 待验证

- C>A 的下游提升能否在 seed0/seed1 复现，以及均值/方差是否足以支持贡献表述。
- SADFFM 是否真正改善几何对应；当前没有真实配准 ground truth 证据。
- 融合低于 IR 的主要因果机制；目前只有置信度、边界困难和轻微定位劣势等关联。
- DSDAM、P2 及其交互在冻结 M3FD 协议上的正式 AP、逐类 AP、小目标 AP/Recall。
- ESSD 增益是否超出增加参数与计算量带来的容量效应。
- 当前最终融合方案相对代表性公开融合方法的同协议表现。
- 最终冻结配置在保留 test 和第二数据集上的泛化。

## 7. 研究设计决策

1. 论文以完整“融合→检测”作为一章，内部消融和外部方法对比都必须做；A/B/C 是
   消融，不是外部对比。
2. 开发顺序坚持“先确认融合是否有任务价值，再验证模块，最后外部对比”，避免在
   内部机制尚未成立时铺开大量昂贵训练。
3. 所有模块必须先写清设计原因、插入位置和可证伪问题，不能从结果倒推故事。
4. SADFFM 暂保留是因为下游 C>A，而不是因为融合 loss 或合成位移成立；最终是否
   宣称贡献取决于多种子和成本。
5. ACG-AW 不因“权重会变化”就算创新；可靠性行为和任务收益未建立，所以降级。
6. 检测端必须做 DSDAM×P2 2×2，而不能只比较 baseline 与 ESSD，否则无法区分 P2
   和 DSDAM。
7. YOLO11s 是正式主模型；YOLO11n 仅用于 smoke，兼顾论文说服力和资源成本。
8. 核心组才做多种子：先 seed42 四组筛选，再仅对 baseline 与最终候选补 seed0/1；
   不重复所有失败方案。
9. 外部方法必须使用公开代码，在同一冻结 manifest 和 detector 设置下重训。未知
   split 的论文数字只能作背景，不能与本项目结果同列直接判优。
10. test 在设计冻结后一次性使用；历史曾接触 ACG-AW/TNO 等测试信息必须披露，
    不能笼统声称项目从未看过任何测试集。
11. 失败实验、布局 bug、无效 probe 都是研究证据，必须保留，防止未来重复浪费。
12. 不为凑三个创新点添加新模块；两个有稳定证据的贡献优于三个未验证命名模块。

### 7.1 论文与导师汇报约束

- 用户要求按顶刊/顶会的可复现流程组织，同时服务于学位论文和阶段性导师汇报。
- 必须区分开发集、验证集和最终测试；必须同时有内部消融与外部方法对比。
- 必须从模块机理和插入位置提出研究问题，不能看到结果后倒推模块理由。
- 现有聊天没有记录导师对某个具体模块的正式书面结论；交接时不得虚构“导师已经
  认可/否定”。如新聊天获得导师新反馈，应记录原话、日期及其对实验顺序的影响。
