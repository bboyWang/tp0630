# 研究进展日志

> 本文档用于记录每次工作的关键决策、代码变更和研究进展，供跨设备同步阅读。
> 另一台机器（10.3.x.x）的 AI 请优先阅读本文档了解最新状态。

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
