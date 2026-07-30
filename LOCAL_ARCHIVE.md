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

*存档人：三楼 AI（opencode） | 首次存档：2026-07-30*
