"""
用 v7 最佳模型对未标注池做 Margin 采样，输出 Top-30 待标注样本
"""
import os, sys, numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, r'E:\tp0630')
from dataset import WetlandDataset, collate_fn
from model import UNet

dev = torch.device('cuda')
C = {'data_dir': r'E:\tp0630', 'in_ch': 26, 'num_cls': 3, 'base_ch': 32,
     'patch': 256, 'bs': 4, 'test_n': 300, 'val_n': 20}

# ── 数据集划分（和 v7 完全一致） ──
torch.manual_seed(42); np.random.seed(42)
ds_base = WetlandDataset(C['data_dir'], patch_size=C['patch'], augment=False, quadrant_mode=False)
all_ids = list(range(len(ds_base)))
np.random.shuffle(all_ids)
test_ids = set(all_ids[:C['test_n']])

labeled_ids = [i for i in range(len(ds_base)) if ds_base.samples[i]['is_human'] and i not in test_ids]
np.random.shuffle(labeled_ids)
val_ids = labeled_ids[:C['val_n']]
train_base_ids = labeled_ids[C['val_n']:]
pool_ids = [i for i in all_ids[C['test_n']:] if i not in train_base_ids and i not in val_ids]

print(f"Pool samples: {len(pool_ids)}")

# ── 加载 v7 最佳模型 ──
model = UNet(C['in_ch'], C['num_cls'], C['base_ch']).to(dev)
ckpt = r'E:\tp0630\checkpoints_v7\best.pth'
if not os.path.exists(ckpt):
    print(f"[ERROR] Checkpoint not found: {ckpt}")
    sys.exit(1)
model.load_state_dict(torch.load(ckpt, weights_only=False))
model.eval()
print(f"Loaded: {ckpt}")

# ── Margin 采样（每个 pool 样本 4 个象限取平均） ──
ds_pool_quad = WetlandDataset(C['data_dir'], patch_size=C['patch'], augment=False, quadrant_mode=True)

all_uncertain = {}
print(f"Scoring {len(pool_ids)} pool samples across 4 quadrants each...")

for i, bid in enumerate(pool_ids):
    quad_scores = []
    for q in range(4):
        idx = bid * 4 + q
        f, m, base, _ = ds_pool_quad[idx]
        f = f.unsqueeze(0).to(dev)
        with torch.no_grad():
            probs = F.softmax(model(f), dim=1)
            t2, _ = torch.topk(probs, 2, dim=1)
            margin = 1.0 - (t2[:, 0] - t2[:, 1])
            quad_scores.append(margin.mean().item())
    avg_margin = np.mean(quad_scores)
    all_uncertain[base] = avg_margin
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(pool_ids)}")

# ── 排序输出 ──
ranked = sorted(all_uncertain.items(), key=lambda x: x[1], reverse=True)
n_select = min(30, len(ranked))

out_file = os.path.join(C['data_dir'], 'uncertain_v7.txt')
with open(out_file, 'w', encoding='utf-8') as f:
    f.write(f"v7 Active Learning — Top {n_select} Uncertain Samples (Margin)\n")
    f.write(f"Scored: {len(ranked)} pool samples (avg over 4 quadrants)\n")
    f.write(f"{'='*60}\n\n")
    for i, (name, score) in enumerate(ranked[:n_select], 1):
        f.write(f"{i:3d}. score={score:.4f}  {name}\n")
    f.write(f"\n{'='*60}\nFull ranked list:\n")
    for i, (name, score) in enumerate(ranked, 1):
        f.write(f"{i:4d}. {score:.4f}  {name}\n")

print(f"\n{'='*55}")
print(f"Top-15 most uncertain (save to uncertain_v7.txt):")
print(f"{'='*55}")
for i, (name, score) in enumerate(ranked[:15], 1):
    print(f"  {i:2d}. score={score:.4f}  {name}")
print(f"\nFull list saved to: {out_file}")
