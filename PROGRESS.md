# 研究进展日志

> 本文档用于记录每次工作的关键决策、代码变更和研究进展，供跨设备同步阅读。
> 另一台机器（10.3.x.x）的 AI 请优先阅读本文档了解最新状态。

---

## 2026-07-31 — v9 完成：训练迁移服务器双4090，全面大幅超越 v8

### 本次工作
- 训练从笔记本 RTX 3050 4GB 迁移至服务器（2×RTX 4090 24GB, /data/wxy/tp0630）
- 完成 v9 训练 + 主动学习打分（全程约 1 小时）
- 数据本地↔服务器全量同步（43.5GB tiff 修复重传）；chip_01505 因质量差删除
- v8-1 批次 29 个新标注样本入账（ready.txt）

### 代码变更
| 文件 | 操作 | 说明 |
|------|------|------|
| `train_v9.py` | 新增 | 服务器版训练脚本（base_ch=64, bs=16, CE+Dice, warmup→cosine, EMA, bf16） |
| `split_v9.json` | 新增 | 钉死的数据划分：Val/Test 与 v8 逐样本一致（01505 幻影补位法复现 v8 shuffle 索引） |
| `ready.txt` | 更新 | 新增 `# v8-1 批次` 29 个样本 |
| `results_v9.txt` | 新增 | v9 训练结果 |
| `uncertain_v9.txt` | 新增 | v9 主动学习排序（pool 1278） |
| `train_v9.log` | 新增 | 完整训练日志 |

### v9 训练结果（Test 集与 v8 完全相同，直接可比）
| 指标 | v8 | **v9** | 提升 |
|------|-----|--------|------|
| Best Val OA | 85.38% (E10) | **92.75%** (E37) | +7.4pt |
| Best Val mIoU | 29.68% | **47.76%** | +18.1pt |
| Test OA | 67.07% | **92.21%** | **+25.1pt** |
| Test mIoU | 22.28% | **50.40%** | **+28.1pt** |
- Early Stop @ E57（patience 20），训练曲线平稳无震荡（v8 曾在 16%~85% 剧烈波动）
- 数据集：Train 147 基础（588 四象限，118+29）| Val 48 | Test 30 | Pool 1278

### 配置（v9 = 硬件解锁版，其余与 v8 一致）
- U-Net base_ch 32→64（13.2M 参数）、batch 2→16、CE+Dice(1:1)
- lr 3e-3（3 epoch warmup → 平滑 cosine，去掉了 v8 的重启）、AdamW wd=1e-4、梯度裁剪 1.0
- EMA(0.999) 双分支评估择优、bf16 autocast、torch 2.4.1+cu121（服务器 conda 环境 torch_cu121）
- 主动学习同口径（四象限平均 Margin），v9 Top-30 分数峰值 0.53（v8 为 0.82，模型明显更自信）

### 观察与备注
- EMA 后期读数剧烈波动且始终未胜 raw，best 来自 raw 分支；v10 可调整 EMA decay 或弃用
- chip_02164 有 PNG 无 JSON，从未进入数据集（不影响）；01505 删除后 ds=1503

### 下一步
1. 标注 `uncertain_v9.txt` Top-30 → `ready.txt` 加 `# v9-1 批次` → v10 训练
2. v10 候选（一次一项）：背景负标签类（最高优先）、NDWI/MNDWI/NDVI 指数通道、90° 旋转、
   3 种子集成分歧采样、伪标签半监督、全图 512 输入

---

## 2026-07-30 — v8 完成，项目首次 GitHub 同步

### 本次工作
- 完成 v8 轮次训练（四象限策略 + Margin Sampling）
- 首次将项目代码和文档同步至 GitHub
- 建立跨设备研究同步机制

### 代码变更
| 文件 | 操作 | 说明 |
|------|------|------|
| `dataset.py` | 新增 | 数据加载器，四象限切分，26波段TIFF+JSON多边形标注 |
| `model.py` | 新增 | 轻量级U-Net（330万参数），适配RTX 3050 4GB |
| `train_active.py` | 新增 | v8主训练脚本，含主动学习Margin Sampling |
| `rank_uncertain.py` | 新增 | v7不确定性排序脚本（独立运行） |
| `ready.txt` | 新增 | 人类标注样本注册表（226个样本） |
| `project_knowledge.txt` | 新增 | 项目完整知识文档 |
| `results_v8.txt` | 新增 | v8训练结果 |
| `v8_summary.txt` | 新增 | v8轮次汇总（已标注+待标注列表） |
| `uncertain_samples.txt` | 新增 | v7主动学习Top30结果 |
| `uncertain_v8.txt` | 新增 | v8主动学习排序结果（1308个pool样本） |
| `TPrf0515.js` | 删除 | 旧实验文件 |
| `mask_to_labelme.py` | 删除 | 旧实验文件 |
| `test_doubao.py` | 删除 | 旧实验文件 |

### 当前研究状态

**v8 训练结果：**
- Best Val OA: 85.38% (Epoch 10)
- Best Val mIoU: 29.68%
- Test OA: 67.07% (30个人类标注样本)
- Test mIoU: 22.28%
- Early Stop @ Epoch 25

**数据集划分：**
- Train: 118个基础样本 → 472个四象限训练样本
- Val: 48个人类标注样本（固定）
- Test: 30个人类标注样本（固定）
- Pool: 1308个未标注样本

**主动学习：**
- v8 Margin Sampling 已产出 Top-30 最不确定样本
- 待人工标注（见 `uncertain_v8.txt` 或 `v8_summary.txt`）

### 下一步
1. 在 LabelMe/EISeg 中标注 v8 Top-30 样本
2. 将30个样本名加入 `ready.txt`，新批次注释 `# v8-1 批次（v8主动学习Top30）`
3. 修改 `train_active.py`：`save_dir` → `checkpoints_v9`，`out_file` → `results_v9.txt`
4. 上传笔记本并运行 v9 训练

### 关键决策
- 四象限策略（512→4×256）是当前最有效的方法，Val OA 从 60% → 90%
- Per-chip Min-Max 归一化 > Z-score（后者导致不收敛）
- AMP 混合精度节省显存，但无法使用 Dice Loss
- Margin Sampling 简单高效，每轮30个样本是合理标注负担

---

## 环境信息

- **开发机（台式机）**: Windows, VSCode, 本项目代码
- **训练机（笔记本）**: RTX 3050 4GB, SSH 10.106.1.196, E:\tp0630\
- **另一台机器（10.3.x.x）**: Windows, VSCode 远程控制同局域网服务器

---

*最后更新: 2026-07-30*
