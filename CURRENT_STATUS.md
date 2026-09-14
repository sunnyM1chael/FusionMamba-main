# 当前研究状态

> - 更新时间：2026-09-14（Asia/Shanghai）
> - 当前分支：`codex/sacafm-essd`
> - 长期背景：见 `PROJECT_CONTEXT.md`

## 当前研究阶段

项目处于“内部开发证据收束 → 检测端正式消融准备”阶段，而不是最终论文测试阶段。

- 模型设计：融合端 A/B/C 和检测端 2×2 结构已经实现，不再无依据新增模块。
- 实验验证：LLVIP 四输入单种子开发对照与融合失败分析已经完成；当前正在冻结 M3FD
  场景级划分，为 DSDAM/P2 正式实验做准备。
- 论文整理：已有可用于组会和方法论说明的报告，但最终主结果、外部对比、多种子和
  test 尚未完成，不能开始写“最终优越性”结论。

## 当前核心问题

### 问题 1：M3FD 正式划分尚未冻结

- 现象：4,200 对 M3FD 数据完整，但当前数据包没有 `scenario.json` 或可追溯的
  train/val/test 成员表。
- 已有证据：官方 TarDAL 代码只表明可由场景范围生成 train/val；官方仓库/当前下载
  包不提供成员表，也没有正式 test 语义。旧 3141/785/274 和 fork 的无 seed 2100/2100
  随机划分均不合格。
- 可能原因：数据发布与论文未把 split 元数据纳入公开版本；连续采集帧若逐图随机分配
  会产生场景泄漏。
- 需要验证：完整 capture/scene 边界、近重复分量、六类和小目标的 group-level 平衡。
- 当前动作：采用自定义 `M3FD-SG-70/10/20-v1`，目标约2940/420/840，组完整性优先。

### 问题 2：融合仍未超过红外单模态

- 现象：LLVIP val AP50:95 为 IR 0.636271，C 0.621695，相差约1.46个百分点。
- 已有证据：C 相比 IR 减少57个FP但增加28个FN；大量 IR-only 目标表现为置信度不足，
  同时共同检测目标的定位略弱；并非等待更多 epoch 或简单改阈值即可解决。
- 可能原因：融合损失无检测反馈，在保留可见纹理与红外目标显著性之间存在取舍；
  边界/困难目标更敏感。以上是机制假设，不是唯一因果结论。
- 需要验证：当前没有值得直接完整重训的新融合机制。固定 IR 残差探针已失败，不应
  继续围绕该配方调 alpha。
- 决策：暂不再做无依据融合重构；先完成检测端创新验证和 C>A 多种子，再决定论文
  如何界定“融合价值”。必须诚实保留 IR 更强的开发结果。

### 问题 3：SADFFM 的贡献强度不足

- 现象：C 的 MSRS fusion loss 比 A 差，但 LLVIP 单 seed AP 高0.549个百分点。
- 已有证据：C>A 是真实的同初始化下游开发结果；C 未胜过 B 的合成位移鲁棒性。
- 可能原因：融合 loss 与目标检测需求不一致；也可能是单 seed 波动或额外容量。
- 需要验证：A/C 至少补 seed0、seed1，统一 evaluator 后报告均值/标准差和成本。
- 决策：不要现在放弃或宣称成立；等检测端 P0 工作完成后只复现核心 A/C。

### 问题 4：检测端 DSDAM 是否有效尚无正式结果

- 现象：baseline/P2-only/DSDAM-only/ESSD 只有 tiny smoke 和初始化审计。
- 已有证据：四结构、权重映射、single-build、隔离验证均通过；M3FD 有20,364个
  resized640 area<32²目标，样本量足够。
- 可能原因：若最终提升存在，可能来自 DSDAM、P2、交互或单纯增加容量。
- 需要验证：冻结 M3FD 上的 2×2 正式消融、逐类与小目标指标、参数/速度/显存。

## 当前实验进展

### 正在进行

- M3FD 全序列场景边界复核。
- 已保存4,199个相邻帧变化值、top80候选、14张每10帧抽样的完整序列概览。
- 已发现58个双模态 equal-dHash 候选组；下一步做全分辨率确认并约束同 split。
- top20变化候选已完成首轮人工分级：9个高置信边界、5个同场景运动、6个不确定；
  记录在 `research/results/m3fd_boundary_review_top20_v1.csv`，尚不能据此冻结完整分组。
- 现有 Codex heartbeat `automation` 每30分钟继续推进；状态无实质变化时保持安静，
  00:00–08:00 不执行工作或启动跨午夜训练。

### 已经完成

- MSRS 权重三组100轮历史开发实验及行为诊断。
- 修复后 MSRS A/B/C 各100轮空间消融与合成位移分析。
- LLVIP A、C、IR、VIS 四组 YOLO11s 各150轮 seed42 开发训练。
- LLVIP 1,200图/3,778框的成对错误、对比度、边缘与案例分析。
- 24×3 CPU 输入干预 v1/v2；stop gate 已触发。
- M3FD 4,200对完整性、34,407框、小目标、尺寸和序列筛查。
- YOLO11s 官方权重上传服务器并通过 SHA-256。
- ESSD 四组共同预训练张量审计在本机和CPU服务器通过。
- 当前代码全测试：19 passed，1 skipped。
- Git 已推送；本交接文档采用的代码/实验基线提交为 `9b508e3`。

### 等待结果/尚未启动

- 没有正在后台运行的正式训练。
- 服务器交接时是无卡模式；不能把无 GPU 状态误报为训练卡住。
- M3FD scene groups、train/val/test manifests 尚未生成。
- M3FD 固定融合图像尚未物化。
- YOLO11s 四组 1-epoch GPU smoke 和正式 seed42 尚未启动。
- A/C 多种子、外部方法、最终 test 尚未启动。

## 下一步计划

### P0 必须完成

#### 1. 冻结 `M3FD-SG-70/10/20-v1`

- 原因：没有可信 split 就训练会导致不可复现和潜在场景泄漏，后续所有正式 AP 无效。
- 动作：完成全序列 contact-sheet 复核；建立 `groups.csv`；将 exact/confirmed-near
  duplicate union 到同一 group；按 group 优化约2940/420/840；报告实际图像数、组数、
  六类框、小目标框、day/night/场景分布；生成 sorted manifests 和 SHA-256；冻结 test。
- 预计：纯 CPU 约1–3小时，取决于人工边界复核，不需要 GPU。

#### 2. 物化一个固定 M3FD 融合输入版本

- 原因：四个 detector 必须看到完全相同的融合图像和标签；不能每组实时重新生成。
- 动作：选择当前冻结的融合 checkpoint/颜色策略；只按 manifest 导出；核对文件数、
  stem、尺寸、解码、标签哈希；记录融合 checkpoint 与代码 commit。
- 预计：需先用少量样本测实际导出速度；GPU约1–3小时范围，不能在无卡模式启动。

#### 3. YOLO11s 四组正式前 GPU smoke

- 原因：虽然 YOLO11n tiny smoke 和 YOLO11s CPU build 已通过，仍需验证 scale-s 的
  CUDA显存、DeformConv backward、loss、保存/重载和最终 evaluator。
- 动作：baseline/P2-only/DSDAM-only/ESSD 各1 epoch，相同小 manifest；记录峰值显存、
  epoch时间、有限性、checkpoint与 evaluator 输出。融合检测正式设置建议保持
  `imgsz640, batch16(显存不够则在所有组统一调整), epochs150, AdamW, lr0=.001,
  lrf=.01, weight_decay=.0005, warmup3, cos_lr, close_mosaic10, patience0, seed42`。
- 精度：当前入口默认四组 AMP 开启、strict deterministic 关闭；若 scale-s GPU smoke
  出现跳步/非有限或实现不兼容，则在正式训练前统一改为四组 FP32。融合前端固定FP32。
- 预计：GPU约20–40分钟。

#### 4. M3FD 2×2 seed42 正式消融

- 原因：这是检测端 DSDAM 创新能否保留的首要研究证据。
- 动作：依次训练四组，禁止并行抢显存；主指标 AP50:95，辅以AP50、Recall、逐类AP、
  预声明 small-object AP/Recall、参数、显存、完整流程延迟。第一组3–5 epoch后按实测
  epoch时长估算剩余时间并调整监控间隔。
- 预计：必须以 smoke 实测估算；不要沿用 LLVIP 3.3–3.8小时/组的时长作为保证。

#### 5. 核心复现与最终 test 纪律

- 原因：单 seed 的0.55个百分点不能支撑稳定创新。
- 动作：seed42筛选后，仅对 baseline 与最终 ESSD 候选补seed0/1；SADFFM 若仍作为
  融合贡献，则 A/C 也补seed0/1。所有选择由 val 完成，test 只在配置冻结后运行一次。

### P1 建议完成

#### 1. 代表性外部方法对比

- 选择有公开代码、可复现的融合方法；使用同一 M3FD/LLVIP frozen manifest、相同
  detector、相同颜色与分辨率策略重新生成和训练。
- 内部 A/B/C、权重组和 DSDAM×P2 是消融；外部不同融合方法是论文主对比，两者都要。

#### 2. 统一最终 evaluator 与结果小数

- A/C/IR/VIS 的训练记录和后续一致重评在末位有约1e-5–1e-4差异。论文表格前用同一
  代码、同一checkpoint、同一参数重评全部组，并以该输出为唯一数字来源。

#### 3. 效率与失败案例

- 报告融合前端、检测器和完整 pipeline 的参数量、显存、batch1 latency、分辨率、
  dtype、warmup次数；对 selective scan/deformable op 未计入 FLOPs 的情况明确说明。
- 固定展示 IR胜C、C胜IR、两者都失败等案例，不按视觉效果挑图。

#### 4. 论文叙事冻结

- 如果 SADFFM 多种子稳定 C>A：表述为“保留任务有用信息的空间自适应融合”，避免
  无真值地声称精确配准。
- 如果不稳定：把它降级为开发探索，不重新命名同一结构制造创新。
- 如果 ESSD 有稳定增益：通过2×2说明 DSDAM、P2和交互；同时承认容量变化。

### P2 可选增强

- 完整 KAIST 官方协议扩展：只有完整数据和官方 miss-rate 评价准备好后再做。
- 第二数据集泛化：在主结果完成后补，不得挤占 P0。
- 960分辨率最终小目标复现：仅当640结果明确正向且GPU预算允许；所有比较组必须一致。
- 更严格真实配准/退化研究：需要明确 ground truth 或预声明受控实验，不以现有合成
  shift 代替真实几何结论。

## 执行与安全边界

- 不删除服务器数据、旧checkpoint或约39GB pilot/约14GB smoke结果，除非用户明确授权。
- `/root/autodl-fs` 文件存储曾约125GB、约20GB空闲，超出免费部分产生约1.04元/日；
  可通过归档/删除可再生数据降低费用，但必须先列出精确目标并获授权。
- 不擅自切换服务器计费/有卡模式，不改本机系统代理，不改关机计划。
- 不重复启动同名训练；启动前检查进程、完成标记、日志和输出目录。
- 每次正式训练保存：代码commit、manifest hash、dataset/annotation provenance、完整参数、
  seed、环境、初始化hash、best/last checkpoint、逐epoch结果和逐样本预测。

## 给下一次聊天的初始化信息

你现在接手这个研究项目，需要知道：研究目标是完成一个“红外/可见光融合后进行目标
检测”的学位论文章节，路线已经确定，不要建议换课题或拆成两章。融合端候选是
SADFFM（代码名SACAFM，Equal weighting），检测端候选是YOLO11s上的DSDAM+P2
ESSD-Head。ACG-AW已经完成开发筛选但没有超过普通Learned gate，不是已成立创新；
固定IR残差probe也已停止。修复后的SADFFM在单seed LLVIP上比无空间A高约0.549个
AP50:95百分点，但仍比IR-only低约1.46个百分点；这个负结果和完整失败分析已经完成，
不要重复做泛泛原因猜测。当前最大问题不是继续改网络，而是冻结可复现、场景不泄漏的
`M3FD-SG-70/10/20-v1`。M3FD有4200对、34407框和20364个resized640小目标，数据
完整但无官方成员表；已生成4199个相邻变化记录、14张序列概览并发现58个equal-dHash
候选组。下一步优先完成groups.csv和manifest审计，然后物化固定融合数据，运行YOLO11s
四组GPU smoke，再做baseline/P2-only/DSDAM-only/ESSD的seed42正式2×2消融。不要在
split冻结前训练，不要把smoke当论文结果，不要用test调参，也不要把自建split称official。
