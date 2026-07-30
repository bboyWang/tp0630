"""
v9 — 服务器双4090版：base_ch 64 + batch16 + CE+Dice + warmup/cosine + EMA
划分钉死：Val/Test 与 v8 完全一致（split_v9.json），v8-1批次29个新样本仅入 Train
"""
import os, sys, json, math, copy, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, '/data/wxy/tp0630')
from dataset import WetlandDataset, collate_fn
from model import UNet, count_params

torch.backends.cudnn.benchmark = True
SMOKE = os.environ.get('SMOKE') == '1'

C = {
    'data_dir': '/data/wxy/tp0630', 'in_ch': 26, 'num_cls': 3, 'base_ch': 64,
    'patch': 256, 'bs': 16, 'epochs': 100, 'patience': 20,
    'base_lr': 3e-3, 'min_lr': 1e-5, 'warmup': 3, 'wd': 1e-4,
    'bf16': True, 'ema_decay': 0.999, 'seed': 42, 'workers': 8,
    'save_dir': '/data/wxy/tp0630/checkpoints_v9',
    'out_file': '/data/wxy/tp0630/results_v9.txt',
    'split_file': '/data/wxy/tp0630/split_v9.json',
    'uncertain_file': '/data/wxy/tp0630/uncertain_v9.txt',
}
if SMOKE:
    C['epochs'] = 2; C['patience'] = 2

torch.manual_seed(C['seed']); np.random.seed(C['seed']); torch.cuda.manual_seed_all(C['seed'])
dev = torch.device('cuda')
print(f"[Device] {dev} | GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
print(f"[Torch] {torch.__version__} | SMOKE={SMOKE}")
os.makedirs(C['save_dir'], exist_ok=True)

# ── 钉死划分：按 split_v9.json 中的样本名分配 ──
with open(C['split_file'], encoding='utf-8') as f:
    SPLIT = json.load(f)
test_names, val_names = set(SPLIT['test']), set(SPLIT['val'])

ds_base = WetlandDataset(C['data_dir'], patch_size=C['patch'], augment=False, quadrant_mode=False)
name2idx = {s['base']: i for i, s in enumerate(ds_base.samples)}
test_ids = sorted(name2idx[n] for n in test_names if n in name2idx)
val_ids = sorted(name2idx[n] for n in val_names if n in name2idx)
human_ids = [i for i in range(len(ds_base)) if ds_base.samples[i]['is_human']]
train_base_ids = sorted(i for i in human_ids if i not in test_ids and i not in val_ids)
pool_ids = [i for i in range(len(ds_base)) if i not in human_ids]
assert len(test_ids) == 30 and len(val_ids) == 48, f"划分异常: test={len(test_ids)} val={len(val_ids)}"
print(f"[Split] Train(base):{len(train_base_ids)} (v8:118 + v8-1新:{len(train_base_ids)-118})  Val:{len(val_ids)}  Test:{len(test_ids)}  Pool:{len(pool_ids)}")

# ── 四象限训练集 ──
ds_train = WetlandDataset(C['data_dir'], patch_size=C['patch'], augment=True, quadrant_mode=True)
train_quad_ids = [bid * 4 + q for bid in train_base_ids for q in range(4)]
ds_val = WetlandDataset(C['data_dir'], patch_size=C['patch'], augment=False, quadrant_mode=False)
print(f"[Train] Quadrant-expanded: {len(train_quad_ids)}")

# ── 模型 ──
model = UNet(C['in_ch'], C['num_cls'], C['base_ch']).to(dev)
mt, _ = count_params(model); print(f"[Model] {mt:,} params (base_ch={C['base_ch']})")
ce_fn = nn.CrossEntropyLoss(ignore_index=-1)

def dice_loss(logits, target, nc=3, eps=1e-6):
    probs = F.softmax(logits.float(), dim=1)
    valid = target != -1
    dl = 0.0
    for c in range(nc):
        p = probs[:, c][valid]
        t = (target[valid] == c).float()
        inter = (p * t).sum()
        dl += 1 - (2 * inter + eps) / (p.sum() + t.sum() + eps)
    return dl / nc

def compute_iou(pred, label, nc=3):
    ious = []
    for c in range(nc):
        p_c = (pred == c); l_c = (label == c)
        inter = (p_c & l_c).float().sum()
        union = (p_c | l_c).float().sum()
        ious.append((inter + 1e-6) / (union + 1e-6))
    return torch.stack(ious).mean()

def lr_at(ep):
    if ep < C['warmup']:
        return C['base_lr'] * (ep + 1) / C['warmup']
    t = (ep - C['warmup']) / max(1, C['epochs'] - C['warmup'])
    return C['min_lr'] + 0.5 * (C['base_lr'] - C['min_lr']) * (1 + math.cos(math.pi * t))

def train_epoch(model, ld, opt):
    model.train(); tl, nb = 0.0, 0
    for f, m, _, _ in tqdm(ld, desc='Train', leave=False):
        f, m = f.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=C['bf16']):
            logits = model(f)
            loss = ce_fn(logits, m) + dice_loss(logits, m)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema_update()
        if not torch.isnan(loss): tl += loss.item(); nb += 1
    return tl / max(nb, 1)

@torch.no_grad()
def evaluate(model, ld):
    model.eval(); vl, vc, vt, vn, ious = 0.0, 0, 0, 0, []
    for f, m, _, _ in ld:
        f, m = f.to(dev), m.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=C['bf16']):
            logits = model(f)
        l = (ce_fn(logits, m) + dice_loss(logits, m)).item()
        if not np.isnan(l): vl += l; vn += 1
        p = logits.argmax(dim=1); valid = m != -1
        vc += (p[valid] == m[valid]).sum().item(); vt += valid.sum().item()
        for i in range(f.size(0)): ious.append(compute_iou(p[i], m[i]).item())
    return vl/max(vn,1), vc/max(vt,1), np.mean(ious) if ious else 0.0

# ── EMA ──
ema_model = copy.deepcopy(model)
for p in ema_model.parameters(): p.requires_grad_(False)
@torch.no_grad()
def ema_update():
    d = C['ema_decay']
    for pe, pm in zip(ema_model.parameters(), model.parameters()):
        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
    for be, bm in zip(ema_model.buffers(), model.buffers()):
        be.copy_(bm)

# ── DataLoader ──
g = torch.Generator(); g.manual_seed(C['seed'])
if SMOKE: train_quad_ids = train_quad_ids[:64]
train_ld = DataLoader(Subset(ds_train, train_quad_ids), C['bs'], shuffle=True,
    collate_fn=collate_fn, pin_memory=True, drop_last=True,
    num_workers=C['workers'], persistent_workers=True, generator=g)
val_ld = DataLoader(Subset(ds_val, val_ids), C['bs'], shuffle=False,
    collate_fn=collate_fn, pin_memory=True, num_workers=4, persistent_workers=True)

opt = AdamW(model.parameters(), C['base_lr'], weight_decay=C['wd'])
ba, bi, bp, bsrc = 0.0, 0.0, os.path.join(C['save_dir'], 'best.pth'), ''
no_imp = 0

print(f"\n{'='*60}")
print(f"v9 | base_ch={C['base_ch']} bs={C['bs']} CE+Dice EMA | {len(train_quad_ids)} train / {len(val_ids)} val / {len(test_ids)} test")
print(f"{'='*60}")

for ep in range(C['epochs']):
    lr = lr_at(ep)
    for pg in opt.param_groups: pg['lr'] = lr
    t0 = time.time()
    tl = train_epoch(model, train_ld, opt)
    vl, voa, viou = evaluate(model, val_ld)
    evl, evoa, eviou = evaluate(ema_model, val_ld)
    tag = ''
    cur_oa, cur_iou, cur_src = (evoa, eviou, 'ema') if evoa >= voa else (voa, viou, 'raw')
    if cur_oa > ba:
        ba, bi, bsrc, no_imp = cur_oa, cur_iou, cur_src, 0
        torch.save((ema_model if cur_src == 'ema' else model).state_dict(), bp)
        tag = f'  -> Best saved ({cur_src}, OA={ba:.4f} mIoU={bi:.4f})'
    else:
        no_imp += 1
    print(f"E{ep+1:3d} | TL:{tl:.4f} VL:{vl:.4f} OA:{voa:.4f} mIoU:{viou:.4f} | EMA_OA:{evoa:.4f} EMA_mIoU:{eviou:.4f} | lr:{lr:.2e} | {time.time()-t0:.1f}s{tag}", flush=True)
    if no_imp >= C['patience']:
        print(f"  Early stop @ E{ep+1}"); break
    if SMOKE and ep >= 1: break

# ── Final Test ──
print(f"\n{'='*60}\nFinal Test ({len(test_ids)} samples, best={bsrc})\n{'='*60}")
model.load_state_dict(torch.load(bp, weights_only=True))
test_ld = DataLoader(Subset(ds_val, test_ids), C['bs'], shuffle=False,
    collate_fn=collate_fn, pin_memory=True, num_workers=4)
_, test_oa, test_miou = evaluate(model, test_ld)
print(f"Test OA: {test_oa:.4f}  Test mIoU: {test_miou:.4f}")
print(f"Best Val OA: {ba:.4f}  Best Val mIoU: {bi:.4f}  (from {bsrc})")

with open(C['out_file'], 'w', encoding='utf-8') as f:
    f.write(f"v9 Supervised Training (Quadrant x4, server 4090, torch {torch.__version__})\n{'='*60}\n")
    f.write(f"Config: base_ch={C['base_ch']} bs={C['bs']} CE+Dice lr={C['base_lr']}(warmup{C['warmup']}->cos) EMA={C['ema_decay']} bf16\n")
    f.write(f"Total human-labeled: {len(human_ids)} (v8:196 + v8-1:29) | Split pinned to v8 (split_v9.json)\n")
    f.write(f"Train base: {len(train_base_ids)} -> {len(train_quad_ids)} quadrants\n")
    f.write(f"Val: {len(val_ids)} (fixed, same as v8)  Test: {len(test_ids)} (fixed, same as v8)\n")
    f.write(f"Best Val OA: {ba:.4f}  Best Val mIoU: {bi:.4f}  (best from: {bsrc})\n")
    f.write(f"Test OA: {test_oa:.4f}  Test mIoU: {test_miou:.4f}\n")

# ── Active Learning: Margin Sampling on pool ──
print(f"\n{'='*60}\nActive Learning — Margin sampling on pool ({len(pool_ids)} samples)\n{'='*60}")
model.eval()
ds_pool_quad = WetlandDataset(C['data_dir'], patch_size=C['patch'], augment=False, quadrant_mode=True)
pool_quad_ids = [bid * 4 + q for bid in pool_ids for q in range(4)]
if SMOKE: pool_quad_ids = pool_quad_ids[:32]
pool_ld = DataLoader(Subset(ds_pool_quad, pool_quad_ids), 64, shuffle=False,
    collate_fn=collate_fn, pin_memory=True, num_workers=C['workers'])

margins = {}
with torch.no_grad():
    for f, _, bases, _ in tqdm(pool_ld, desc='Margin', leave=False):
        f = f.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=C['bf16']):
            probs = F.softmax(model(f), dim=1)
        t2, _ = torch.topk(probs.float(), 2, dim=1)
        margin = (1.0 - (t2[:, 0] - t2[:, 1])).mean(dim=(1, 2))
        for b, mg in zip(bases, margin.cpu().tolist()):
            margins.setdefault(b, []).append(mg)

avg = {b: float(np.mean(v)) for b, v in margins.items()}
ranked = sorted(avg.items(), key=lambda x: x[1], reverse=True)
n_select = min(30, len(ranked))
with open(C['uncertain_file'], 'w', encoding='utf-8') as f:
    f.write(f"v9 Active Learning — Top Uncertain Samples\n{'='*55}\n")
    f.write(f"Scored {len(ranked)} pool samples (average margin across 4 quadrants)\n\n")
    f.write(f"{'Rank':<5} {'Score':<8} Sample\n" + '-' * 60 + '\n')
    for i, (name, score) in enumerate(ranked[:n_select], 1):
        f.write(f"{i:<5} {score:<8.4f} {name}\n")
    f.write(f"\nFull ranked list:\n")
    for i, (name, score) in enumerate(ranked, 1):
        f.write(f"{i:4d}. {score:.4f}  {name}\n")

print(f"\nTop-10 most uncertain:")
for i, (name, score) in enumerate(ranked[:10], 1):
    print(f"  {i:2d}. score={score:.4f} {name}")
print(f"\n[Done] Model:{bp}  Results:{C['out_file']}  Uncertain:{C['uncertain_file']}")
