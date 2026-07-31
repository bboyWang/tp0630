# 本地操作存档日志（三楼台式机）

> 本文档用于存档**本机（三楼台式机，C:\Users\18264\Desktop\tp0630）上的每一次 AI 对话操作**，
> 每次会话结束后追加新条目并 `git commit + push` 同步至云端。
> 规则：只追加、不改写历史条目；每条记录包含：做了什么、删了什么、改了什么、待办问题。

---

## 2026-07-30 — 首次 GitHub 同步 + 本地大扫除 + 同步机制建立

### 1. 完整阅读 GitHub 仓库
- 仓库：https://github.com/bboyWang/tp0630 （已公开）
- 已通读全部内容：`PROGRESS.md`、`REMOTE_SETUP.md`、`SETUP.md`、`project_knowledge.txt`、
  `v8_summary.txt`、`results_v8.txt`、`ready.txt`、`dataset.py`、`model.py`、`train_active.py`、
  `rank_uncertain.py`、`uncertain_samples.txt`、`uncertain_v8.txt`、`.gitignore`
- 理解要点：青藏高原湿地 3 类语义分割（湿地/水体/湖泊），26 波段 512×512 TIFF，
  轻量 U-Net 3.3M 参数，四象限策略（512→4×256）为关键创新，主动学习 Margin Sampling
  每轮 Top-30 送标注。v8 已完成：Best Val OA 85.38%、Test OA 67.07%。
  下一步 v9：标注 `uncertain_v8.txt` Top-30 → 加入 `ready.txt`（`# v8-1 批次`）→ 改
  `train_active.py` 的 `save_dir=checkpoints_v9`、`out_file=results_v9.txt` → 训练。

### 2. 从 GitHub 复制到本地的文件（6 个）
- `PROGRESS.md`、`REMOTE_SETUP.md`、`SETUP.md`、`project_knowledge.txt`、`v8_summary.txt`（本地原本缺失）
- `uncertain_samples.txt`（用仓库版覆盖本地旧版；本地旧版是"5 轮 100 样本"的远古主动学习记录，
  其信息已被 `ready.txt` 批次记录和 `v8_summary.txt` 完整取代，无信息损失）

### 3. 删除的废物文件（30 个 + __pycache__，均不触碰数据集与权重）
- **官方已移除的旧实验文件**：`TPrf0515.js`、`mask_to_labelme.py`
- **旧训练日志**（v6 及更早，约 1.9MB）：`train_log.txt`、`finetune_log.txt`、
  `train_v3.log`、`train_v4.log`、`train_v5.log`、`train_v6.log`、`train_v6b.log`
- **一次性诊断/调试脚本**：`check_dir/files/deps/bands/label/size.py`（6 个）、
  `test_cuda.py`、`test_dataset.py`、`test_v3.py`、`debug_v3.py`
- **旧规划/旧结果/旧主动学习产物**：`plan_v3.txt`、`plan_v4.txt`、`results_v6.txt`、
  `results_v6b.txt`、`uncertain_v3.txt`、`uncertain_for_annotation.txt`、`uncertain_to_label.txt`
- **已废弃方案产物**：`band_stats.npz` + `compute_stats.py`（Z-score 方案已弃用）、
  `count_labels.py`、`list_samples.py`
- `__pycache__/`（Python 缓存，自动再生成）

### 4. 保留未动的内容
- **数据集（绝不动）**：`tiff/`（43.5GB，1526 文件）、`eiseg_ready/`（1.6GB，4535 文件）
- **当前代码与文档**：`dataset.py`、`model.py`、`train_active.py`、`rank_uncertain.py`、
  `ready.txt`、`samples.txt`、`results_v7.txt`、`results_v8.txt`、`uncertain_v7.txt`、
  `uncertain_v8.txt`、`v8_summary.txt`、`run_v8.bat`、`train_v7.log`、`train_v8.log`
- **全部模型权重**：`checkpoints/`（88.7MB，远古 round1-5+finetune）、
  `checkpoints_v3/v4/v5/v6/v6b/v7/v8/`
- **待用户定夺的"次废物"（暂未删除，共约 234MB）**：
  - `checkpoints/`（88.7MB，远古 round1-5 + finetune 权重，对应已删的 finetune_log 时代）
  - `checkpoints_v3/`（25.3MB）、`checkpoints_v4/`（63.3MB）、`checkpoints_v5/`（38MB）旧权重
  - `mask/`（18.6MB，1529 个旧栅格标签，现行代码直接从 JSON 栅格化，不再使用）

### 5. 环境决策（用户拍板）
- 本机（三楼）**不安装任何深度学习环境**——只是中转站，最多跑 LabelMe 标注。
- 训练通过本机 VSCode 的 Remote-SSH 连接 **10.3.1.253（Linux 服务器）**进行；
  数据后续可能也需同步到该服务器。
- 本机已有：Git 2.55.0；无 conda、无 Python、无 NVIDIA 显卡（仅 Intel UHD 730 核显）。

### 6. 建立的机制
- 本地 `tp0630` 已初始化为 git 仓库，关联 `origin = https://github.com/bboyWang/tp0630.git`，
  分支 `master` 跟踪 `origin/master`。
- `.gitignore` 已扩充：`tiff/`、`eiseg_ready/`、`mask/`、`checkpoints*/`、`*.npz`、`*.log`、
  `desktop.ini` 全部忽略，确保 43GB 数据集永远不会被误传云端。
- 新建本存档文件 `LOCAL_ARCHIVE.md`，此后**每次会话操作都会追加记录并推送云端**。

### 7. 待办 / 待用户确认
- [ ] 用户确认是否删除第 4 节列出的"次废物"（旧权重 + mask/，约 234MB）
- [ ] LabelMe 安装位置的分析结论（见对话；建议装在本机 Windows 上，而非 Linux 服务器）
- [ ] 数据同步至 10.3.1.253 服务器的方式与时间
- [ ] v9 准备工作：标注 v8 Top-30（见 `uncertain_v8.txt` / `v8_summary.txt`）

---

## 2026-07-30（同日第二次）— LabelMe 安装 + 服务器全量同步完成

### 1. 用户决策更新
- 旧权重（checkpoints/、checkpoints_v3/v4/v5 等）**全部保留**，不删除。
- `mask/`（GEE 下载的老分类文件）**保留**，以后可能要看。
- 标注工作由用户手动完成；本机仅需装好 LabelMe。

### 2. LabelMe 安装完成（本机）
- 安装 Miniconda → `C:\Users\18264\miniconda3`（conda 26.5.3，已接受 ToS）。
- 新建虚拟环境 **`labelme`**（python=3.10），`pip install labelme` → **labelme 6.3.1**（PyQt5 5.15.11）。
- 桌面已创建启动器：**`启动LabelMe.bat`**（双击即用）。
- 验证：`import labelme` OK。未安装任何深度学习框架（符合"中转站"定位）。

### 3. 服务器连接（dell@10.3.1.253）
- 服务器：dell-PowerEdge-T640（Linux）；SSH 用户 `dell`，密码方式登录。
- 工具：`plink.exe`/`pscp.exe`（PuTTY 官方）放在 `C:\Users\18264\AppData\Local\Temp\opencode\tools\`，
  若被系统清理，从 https://the.earth.li/~sgtatham/putty/latest/w64/ 重新下载即可（各约 1MB）。
- hostkey 指纹：`SHA256:V2H3mm56LTwgYdMmuT+gQEGYYFmXxM8Ucm6hqyD4/34`。
- **纪律：只操作 `/data/wxy/tp0630`，wxy 目录的同级/上级文件夹一律不碰（别人的数据）。**

### 4. 服务器清理 + 文档同步
- 远程删除与本地一致的 30 个废物文件 + `__pycache__`。
- 上传 8 个新/更新文件：`PROGRESS.md`、`REMOTE_SETUP.md`、`SETUP.md`、`project_knowledge.txt`、
  `v8_summary.txt`、`LOCAL_ARCHIVE.md`、`.gitignore`、`uncertain_samples.txt`。

### 5. 数据同步（重要发现与修复）
- **发现**：服务器 02:26 的那次同步已中断——tiff 里 1524/1526 个文件是 256/512KB 的残缺桩，
  images 里 2270/3009 个文件不完整。
- **修复**：全量重传 `tiff/`（43.5GB，约 46MB/s，约 16 分钟）+ `eiseg_ready/images/`（1.6GB）。
- **最终校验（逐文件比对大小）**：
  | 目录 | 文件数 | 缺失 | 大小不符 |
  |---|---|---|---|
  | tiff/ | 1526 | 0 | 0 |
  | eiseg_ready/images/ | 3009 | 0 | 0 |
  | eiseg_ready/labels/ | 1526 | 0 | 0（原本就完整） |
  | mask/ | 1529 | 0 | 0（原本就完整） |
- 本地与服务器数据现已**完全一致**。

### 6. 服务器磁盘预警
- `/data` 分区 6.2TB，同步后**仅剩约 50GB（100% 水位）**。本分区是多用户共享，
  后续大数据写入（如 v9 checkpoints）需留意；建议有机会时提醒管理员或清理。

### 7. 持续同步机制（此后每次会话执行）
1. 本地操作完成后：追加本存档 → `git commit` → `git push`（GitHub）。
2. 有变动的文件：用 `pscp` 同步到服务器 `/data/wxy/tp0630/` 对应位置。
3. 新增标注 JSON（用户手动标注产物）也需要在会话结束时同步到服务器 `eiseg_ready/images/`。

---

## 2026-07-30（同日第三次）— v8-1 批次标注完成核验 + 服务器 GPU 确认 + v9 筹备

### 1. 标注成果核验（用户手动完成）
- v8 Top-30 中 **29 个已有 shapes 标注**：26 个为 07-29 17:47 ~ 07-30 22:22 新标注；
  3 个（chip_02152、02168、02167）JSON 里是 07-16 的旧 shapes（各 1 个），**待用户确认是否需重标**；
  1 个（chip_01505_lc1_G29_09）**本地无 PNG/JSON**（当年未导出预览图，无法标注），本轮跳过。
- 另有 6 个旧标注被精修（chip_01025、01183、01519、01891、02153、02169，均属 === 批次）。

### 2. 同步完成
- 32 个变动 JSON 已上传服务器 `eiseg_ready/images/`。
- `ready.txt` 已新增 **`# v8-1 批次（v8主动学习Top30）`，26 个样本**（保守起见未含 3 个旧 shape 存疑样本），
  已同步服务器；GitHub 本地已 commit（`83c9224`），**push 因网络连不上 GitHub 暂缓，下次会话重试**。

### 3. 服务器硬件/环境确认（dell-PowerEdge-T640）
- **GPU：2× NVIDIA RTX 4090 24GB**（驱动 535.183.01，CUDA 12.2，当前空闲无进程）
- CPU 40 核，内存 62GB；系统 Python 3.8.10，`~/miniconda3` 存在（未进 PATH），**未装 PyTorch**，
  系统 pip 有 rasterio 1.3.11。
- 结论：训练可从笔记本 RTX 3050 4GB 迁移到服务器，性能余量巨大。

### 4. v9 实验新方案要点（详见对话记录）
- batch 2→16、base_ch 32→64、可上 Dice/Focal Loss（摆脱 4GB 显存的 AMP 限制）、
  可加 90° 旋转增强、NDWI/NDVI 指数通道、可选 DataParallel 双卡、EMA/TTA、更久 early stop。
- 需在服务器建 conda 环境（torch cu121，驱动 CUDA 12.2 兼容）+ 改代码路径为 `/data/wxy/tp0630`（Linux）。
- **待用户拍板后执行**：服务器装环境 → 改 v9 配置 → 首次双卡训练。

### 5. 负标签（非湿地）讨论结论（详见对话）
- 建议：不必全量标背景；精选 30~50 张做"全要素稠密标注"（含背景类）作为高质量 Val/Test，
  训练集仍用稀疏正例 + 加背景类通道，性价比最高。

---

## 2026-07-30（同日第四次）— 标注收尾：29/29 全部入账 + 01505 删除 + GitHub 推送修复

### 1. ready.txt 最终状态
- 3 个存疑样本（chip_02152、02168、02167）经用户确认为**标注准确**，已补录进 `# v8-1 批次`。
- **v8-1 批次现共 29 个样本**（v8 Top-30 减去被删的 01505），本地+服务器+GitHub 三处已同步（commit `517d030`）。

### 2. chip_01505 删除（用户决定：样本质量太差）
- 本地 tiff 与服务器 tiff 中的 `chip_01505_lc1_G29_09_tp0518_features.tif` **均已删除**（两边 tiff 各剩 1525 个）。
- 该样本本就无 PNG/JSON，不影响 dataset.py；`uncertain_v8.txt` 中的历史排名记录保留备查。

### 3. GitHub 推送故障修复（重要经验）
- 现象：git push 报 "Could not connect to server"，但浏览器能开 GitHub。
- 原因：本机有系统代理 `127.0.0.1:7897`（浏览器走代理），git 未配置代理，直连在 TLS 握手时被重置。
- 修复：`git config --global http.https://github.com.proxy http://127.0.0.1:7897`
  （**仅对 github.com 生效**，不影响其他 git 操作；若代理软件关闭导致 push 失败，先检查此项）。
- 修复后 4 个提交（`83c9224`、`34f45ea`、`517d030` 等）已全部推送成功。

### 4. 当前全景
- 数据：本地 = 服务器（tiff 1525 / images 3009 / labels 1526 / mask 1529），GitHub 文档最新。
- v9 训练方案已定（服务器双 4090、batch=16、base_ch=64、CE+Dice、指数通道），**用户指示暂缓开工**。

---

## 2026-07-31 — v9 服务器首训成功：Test OA 92.21% / mIoU 50.40%（大幅超越 v8）

### 1. 服务器环境（新建）
- conda 环境 `torch_cu121`（python 3.10）：**torch 2.4.1+cu121** + torchvision/audio + rasterio/numpy/tqdm/pillow
- 说明：服务器直连外网慢，但 pip 已预置清华镜像（/etc/pip.conf），几分钟装完；
  驱动 CUDA 12.2 不支持 cu124，PyPI 的 torch 2.4.1 默认即 cu121，与文档的 2.5.1 API 兼容。
- 服务器的 7892 代理未运行（.bashrc 里有 watch_proxy，需要时用户手动开）。

### 2. 划分钉死（关键工程）
- 发现本地/服务器数据与 v8 时代差 1 个样本（01505 被删，ds 1504→1503）；
  用"幻影补位法"把 01505 作为 pool 幻影插回排序位置，精确复现 v8 的 shuffle 索引，
  生成 `split_v9.json`（test 30 / val 48 与 v8 逐样本一致，train 118）。
- chip_02164 有 PNG 无 JSON，从未进过数据集（v8 pool 名单中也无它），无影响。

### 3. v9 训练（train_v9.py，pid 24946，约 1 小时）
- 配置：base_ch=64（13,208,323 参数）、bs=16、CE+Dice、lr 3e-3 warmup3→cosine、
  EMA 0.999、bf16、100ep/patience20、种子 42、单卡 4090。
- 轨迹：E1 85.88/27.30 → E8 89.58/37.89 → E11 90.60/44.33 → E20 92.60/43.10 →
  **E37 最佳 92.75/47.76** → E57 早停。
- **最终：Test OA 92.21%、Test mIoU 50.40%**（v8：67.07%/22.28%，+25.1pt/+28.1pt）。
- EMA 后期读数剧烈波动始终未胜 raw，best=raw；v10 再调。
- AL 同口径打分完成（pool 1278）：`uncertain_v9.txt`，Top1 score 0.5258（模型比 v8 自信得多）。

### 4. 产物归位（三处一致）
- 服务器→本地：results_v9.txt、uncertain_v9.txt、train_v9.log、checkpoints_v9/best.pth（51.6MB）
- 文档：PROGRESS.md 已加 v9 条目；本存档同步更新；GitHub+服务器同步。

### 5. 待办
- [ ] 用户标注 v9 Top-30（uncertain_v9.txt）→ ready.txt 加 `# v9-1 批次` → v10
- [ ] v10 候选：背景负标签类（优先）、指数通道、90°旋转、集成分歧、伪标签

---

## 2026-07-31（同日第二次）— V10_PLAN.md 成型 + 波段分析完成 + 标注期启动

### 1. TIFF 波段元数据确认
- 读取服务器 TIFF 元数据，确认 30 波段结构：
  - 1-10: S2 光学(blue~swir2) + **NDVI/NDWI/MNDWI/EVI**（E 期）
  - 11: S2CNT_E（丢弃）| 12-21: S2 光学+指数（L 期）| 22: S2CNT_L（丢弃）
  - 23-24: DEM/SLOPE | 25-26: VV/VH_E | 27: S1CNT_E（丢弃）| 28-29: VV/VH_L | 30: S1CNT_L（丢弃）
  - **NDVI/NDWI/MNDWI/EVI 四个指数双时相共 8 个波段已在输入中**——无需额外添加指数通道。
- 4 个丢弃的计数波段完全正确。

### 2. V10_PLAN.md 完成
- 详细记录 v4→v9 全部研究进展、配置、结果、教训（含数据表格）
- v10 方案：背景负标签 + 90°旋转增强 + 3模型分歧AL + 伪标签半监督（Mean Teacher）
- ③+④ 的学术创新链条（分歧→人标 / 一致→机标闭环）作为论文核心框架
- 标注任务清单 + 代码改动清单 + 硬件时间预估
- 用户已保存，用于明日汇报。

### 3. 当前标注阶段
- 用户正在标注：v9 Top-30（30张）+ 背景负标签（30~50张稠密全要素标注）
- 标注完成后执行 v10 代码实现与训练。

---

## 2026-07-31（同日第三次）— 汇报 PPT 生成 + QA + 环境补装

### 1. 生成 PPT（12 页）
- 输出：`湿地语义分割研究进展_v4-v9_20260731.pptx`（378KB，16:9，微软雅黑）
- 生成脚本：`C:\Users\18264\AppData\Local\Temp\opencode\pptbuild\gen_ppt.js`（pptxgenjs 本地安装）
- 12 页结构：标题 / 背景任务 / 数据资源30波段 / 核心方法(四象限+AL) /
  技术演进v4→v9 / 核心结果柱状图 / v9训练解析 / 关键教训 /
  V10四大改动 / 人机协同标注闭环 / V10预期收益与计划 / 总结展望
- 关键数据均已核实与 v8/v9 结果一致：Test OA 92.21%(+25.1pt)、mIoU 50.40%(+28.1pt)、
  图表 v7/v8/v9 三系列 4 指标。

### 2. QA（本模型不支持看图，采用程序化 QA，全部通过）
- 本机补装：python-pptx 1.0.2 + markitdown（labelme env）；LibreOffice 26.2.5（winget）
- soffice 转 PDF/PNG：12 页全部渲染成功，图表柱状矢量图形确认存在
- PyMuPDF block 级检查：0 越界、0 溢出（全部文本块在容器形状内）
- 逐页文本抽取核验：12 页内容完整，`→`/`↺`/`✓` 符号渲染正常
- 字形碰撞排查：char 级重叠为 LibreOffice 导出"推进宽度盒"假阳性，
  布局代码核实 statCard 值/标签框留 0.06in 空隙、流程框 0.42in 间隙，真实无碰撞
- 修正环境副作用：markitdown 把 onnxruntime 降到 1.20.1 破坏 osam，已还原为 1.23.2

### 3. 待办（与上次一致，无新增阻塞）
- [ ] 用户标注 v9 Top-30（uncertain_v9.txt）→ ready.txt 加 `# v9-1 批次` → v10
- [ ] v10 实现：背景负标签 + 90°旋转 + 3模型分歧AL + 伪标签半监督（标注完成后启动）

---

*存档人：三楼 AI（opencode） | 首次存档：2026-07-30*
