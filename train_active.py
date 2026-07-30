"""
v8 — 监督训练 + 四象限数据增强 (512→4×256)
所有218人类标注样本，8:2 train/val，固定30人类标注test集
"""
import os, sys, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, r'E:\tp0630')
from dataset import WetlandDataset, collate_fn
from model import UNet, count_params

torch.backends.cudnn.benchmark = True

C = {
    'data_dir': r'E:\tp0630', 'in_ch': 26, 'num_cls': 3, 'base_ch': 32,
    'patch': 256, 'bs': 2, 'epochs': 60, 'lr': 1e-3, 'wd': 1e-4,
    'amp': True, 'seed': 42,
    'save_dir': r'E:\tp0630\checkpoints_v8',
    'out_file': r'E:\tp0630\results_v8.txt',
    'test_n': 30, 'val_n': 48,
}

torch.manual_seed(C['seed']); np.random.seed(C['seed'])
dev = torch.device('cuda')
print(f"[Device] {dev} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
os.makedirs(C['save_dir'], exist_ok=True)

# ── Dataset Split (base, no quadrant) ──
ds_base = WetlandDataset(C['data_dir'], patch_size=C['patch'], augment=False, quadrant_mode=False)

# All human-labeled samples
human_all = [i for i in range(len(ds_base)) if ds_base.samples[i]['is_human']]
np.random.shuffle(human_all)

# Fixed test set: 30 human-labeled only
test_ids = set(human_all[:C['test_n']])
# Remaining human-labeled for train/val split
human_rest = human_all[C['test_n']:]

print(f"[Data] Total:{len(ds_base)}  Human-labeled:{len(human_all)}  Test(fixed):{len(test_ids)}")

# Validation: 48 from remaining human-labeled
val_ids = human_rest[:C['val_n']]
# Train: ALL remaining human-labeled
train_base_ids = human_rest[C['val_n']:]
# Pool = all non-human-labeled samples
pool_ids = [i for i in range(len(ds_base)) if i not in human_all]

print(f"[Split] Train(base):{len(train_base_ids)}  Val(fixed):{len(val_ids)}  Pool:{len(pool_ids)}")

# ── Quadrant-expanded training set ──
ds_train = WetlandDataset(C['data_dir'], patch_size=C['patch'], augment=True, quadrant_mode=True)
train_quad_ids = []
for bid in train_base_ids:
    for q in range(4):
        train_quad_ids.append(bid * 4 + q)
print(f"[Train] Quadrant-expanded: {len(train_quad_ids)} (from {len(train_base_ids)} base)")

# Validation (no quadrants, center crop)
ds_val = WetlandDataset(C['data_dir'], patch_size=C['patch'], augment=False, quadrant_mode=False)

# ── Model ──
model = UNet(C['in_ch'], C['num_cls'], C['base_ch']).to(dev)
mt, _ = count_params(model); print(f"[Model] {mt:,} params")
ce_fn = nn.CrossEntropyLoss(ignore_index=-1)

def compute_iou(pred, label, nc=3):
    ious = []
    for c in range(nc):
        p_c = (pred == c); l_c = (label == c)
        inter = (p_c & l_c).float().sum()
        union = (p_c | l_c).float().sum()
        ious.append((inter + 1e-6) / (union + 1e-6))
    return torch.stack(ious).mean()

def train_epoch(model, ld, opt, sc):
    model.train(); tl, nb = 0.0, 0
    for f, m, _, _ in tqdm(ld, desc='Train', leave=False):
        f, m = f.to(dev), m.to(dev); opt.zero_grad()
        if C['amp']:
            with autocast(): logits = model(f); loss = ce_fn(logits, m)
            sc.scale(loss).backward(); sc.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            sc.step(opt); sc.update()
        else:
            logits = model(f); loss = ce_fn(logits, m)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if not torch.isnan(loss): tl += loss.item(); nb += 1
    return tl / max(nb, 1)

@torch.no_grad()
def evaluate(model, ld):
    model.eval(); vl, vc, vt, vn, ious = 0.0, 0, 0, 0, []
    for f, m, _, _ in ld:
        f, m = f.to(dev), m.to(dev); logits = model(f)
        l = ce_fn(logits, m).item()
        if not np.isnan(l): vl += l; vn += 1
        p = logits.argmax(dim=1); valid = m != -1
        vc += (p[valid] == m[valid]).sum().item(); vt += valid.sum().item()
        for i in range(f.size(0)): ious.append(compute_iou(p[i], m[i]).item())
    return vl/max(vn,1), vc/max(vt,1), np.mean(ious) if ious else 0.0

# ── Training Loop ──
train_ld = DataLoader(Subset(ds_train, train_quad_ids), C['bs'], shuffle=True,
    collate_fn=collate_fn, pin_memory=True, drop_last=True)
val_ld = DataLoader(Subset(ds_val, val_ids), C['bs'], shuffle=False,
    collate_fn=collate_fn, pin_memory=True)

opt = AdamW(model.parameters(), C['lr'], weight_decay=C['wd'])
sch = CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2)
sc = GradScaler() if C['amp'] else None
ba, bi, bp = 0.0, 0.0, os.path.join(C['save_dir'], 'best.pth')
no_imp = 0

print(f"\n{'='*55}")
print(f"v8 Supervised | Quadrant x4 | {len(train_quad_ids)} train / {len(val_ids)} val / {len(test_ids)} test")
print(f"{'='*55}")

for ep in range(C['epochs']):
    tl = train_epoch(model, train_ld, opt, sc)
    sch.step()
    vl, voa, viou = evaluate(model, val_ld)
    print(f"E{ep+1:3d} | TL:{tl:.4f} VL:{vl:.4f} OA:{voa:.4f} mIoU:{viou:.4f}")
    if voa > ba:
        ba, bi, no_imp = voa, viou, 0
        torch.save(model.state_dict(), bp)
        print(f"  -> Best saved (OA={ba:.4f} mIoU={bi:.4f})")
    else:
        no_imp += 1
        if no_imp >= 15: print(f"  Early stop"); break

# ── Final Test ──
model.load_state_dict(torch.load(bp, weights_only=False))
print(f"\n{'='*55}")
print(f"Final Test ({len(test_ids)} independent samples)")
print(f"{'='*55}")
test_ld = DataLoader(Subset(ds_val, list(test_ids)), C['bs'], shuffle=False,
    collate_fn=collate_fn, pin_memory=True)
_, test_oa, test_miou = evaluate(model, test_ld)
print(f"Test OA: {test_oa:.4f}  Test mIoU: {test_miou:.4f}")
print(f"Best Val OA: {ba:.4f}  Best Val mIoU: {bi:.4f}")

# ── Save ──
with open(C['out_file'], 'w', encoding='utf-8') as f:
    f.write(f"v8 Supervised Training (Quadrant x4)\n{'='*55}\n")
    f.write(f"Total human-labeled: {len(human_all)} (116 chip<=377 + 14 -- + 13 = + 45 == + 30 ===)\n")
    f.write(f"Train base: {len(train_base_ids)} -> {len(train_quad_ids)} quadrants\n")
    f.write(f"Val: {len(val_ids)} human-labeled (fixed, center crop)\n")
    f.write(f"Test: {len(test_ids)} human-labeled (fixed, center crop)\n")
    f.write(f"Best Val OA: {ba:.4f}  Best Val mIoU: {bi:.4f}\n")
    f.write(f"Test OA: {test_oa:.4f}  Test mIoU: {test_miou:.4f}\n")

print(f"\n[Done] Model:{bp}  Results:{C['out_file']}")

# ── Active Learning: sample from pool ──
print(f"\n{'='*55}")
print(f"Active Learning — Margin sampling on pool ({len(pool_ids)} samples)")
print(f"{'='*55}")

# Load best model for inference
model.load_state_dict(torch.load(bp, weights_only=False))
model.eval()

# Use pool dataset (no quadrant, center crop for consistent scoring)
ds_pool = WetlandDataset(C['data_dir'], patch_size=C['patch'], augment=False, quadrant_mode=False)
pool_ld = DataLoader(Subset(ds_pool, pool_ids), C['bs'], shuffle=False,
    collate_fn=collate_fn, pin_memory=True)

# Margin scoring (average over 4 quadrants for each pool sample)
margin_scores = {}
with torch.no_grad():
    for f, m, bases, _ in tqdm(pool_ld, desc='Margin', leave=False):
        # f is already a quadrant crop (center crop since augment=False)
        # We need to score on all 4 quadrants and average
        pass  # Will handle below

# Better approach: score each pool sample using the quadrant-expanded dataset
ds_pool_quad = WetlandDataset(C['data_dir'], patch_size=C['patch'], augment=False, quadrant_mode=True)
all_uncertain = {}

for bid in pool_ids:
    # Get 4 quadrants for this pool sample
    quad_scores = []
    for q in range(4):
        idx = bid * 4 + q
        f, m, _, _ = ds_pool_quad[idx]
        f = f.unsqueeze(0).to(dev)
        with torch.no_grad():
            probs = F.softmax(model(f), dim=1)
            t2, _ = torch.topk(probs, 2, dim=1)
            margin = 1.0 - (t2[:, 0] - t2[:, 1])
            quad_scores.append(margin.mean().item())
    # Average margin across 4 quadrants
    avg_margin = np.mean(quad_scores)
    base_name = ds_pool.samples[bid]['base']
    all_uncertain[base_name] = avg_margin
    if (bid - pool_ids[0]) % 100 == 0:
        print(f"  Scored {bid - pool_ids[0] + 1}/{len(pool_ids)}")

ranked = sorted(all_uncertain.items(), key=lambda x: x[1], reverse=True)
n_select = min(30, len(ranked))

# Save uncertain samples
out_uncertain = os.path.join(C['data_dir'], 'uncertain_v8.txt')
with open(out_uncertain, 'w', encoding='utf-8') as f:
    f.write(f"v8 Active Learning — Top Uncertain Samples\n{'='*55}\n")
    f.write(f"Scored {len(ranked)} pool samples (average margin across 4 quadrants)\n\n")
    f.write(f"{'Rank':<5} {'Score':<8} Sample\n")
    f.write('-' * 60 + '\n')
    for i, (name, score) in enumerate(ranked[:n_select], 1):
        f.write(f"{i:<5} {score:<8.4f} {name}\n")
    if len(ranked) > n_select:
        f.write(f"\n... and {len(ranked) - n_select} more (see full list below)\n")
        f.write(f"\nFull ranked list:\n")
        for i, (name, score) in enumerate(ranked, 1):
            f.write(f"{i:4d}. {score:.4f} {name}\n")

print(f"\nTop-10 most uncertain:")
for i, (name, score) in enumerate(ranked[:10], 1):
    print(f"  {i:2d}. score={score:.4f} {name}")
print(f"\nFull list saved to: {out_uncertain}")
