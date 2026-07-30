"""
Dataset v3: 26-band TIFF + JSON labels (30→26, drop counting bands)
Bands removed: S2CNT_E(11), S2CNT_L(22), S1CNT_E(27), S1CNT_L(30)
Keep bands: [0..9, 11..20, 22..25, 27..28] (0-based)
"""
import os, json, re, numpy as np, torch
from torch.utils.data import Dataset
import rasterio
from PIL import Image, ImageDraw

LABEL_MAP = {'湿地': 0, '水体': 1, '湖泊': 2}

# Band indices to KEEP (0-based, from 30 bands)
# Drop: 10(S2CNT_E), 21(S2CNT_L), 26(S1CNT_E), 29(S1CNT_L)
KEEP_BANDS = list(range(10)) + list(range(11, 21)) + list(range(22, 26)) + list(range(27, 29))

class WetlandDataset(Dataset):
    def __init__(self, data_dir, patch_size=256, augment=False, quadrant_mode=False):
        self.data_dir = data_dir; self.patch_size = patch_size; self.augment = augment
        self.quadrant_mode = quadrant_mode  # If True, split 512→4×256 quadrants
        self.img_dir = os.path.join(data_dir, 'eiseg_ready', 'images')
        self.tiff_dir = os.path.join(data_dir, 'tiff')
        self.json_dir = os.path.join(data_dir, 'eiseg_ready', 'images')  # JSONs mixed with PNGs

        # Load human-labeled set from ready.txt
        self.human_set = set()
        ready_file = os.path.join(data_dir, 'ready.txt')
        if os.path.exists(ready_file):
            with open(ready_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    name = re.sub(r'[=#\-]+$', '', line).strip()
                    if name.startswith('QTP_'): self.human_set.add(name)

        self.samples = []
        for f in sorted(os.listdir(self.img_dir)):
            if not f.endswith('.png'): continue
            base = f.replace('.png', '')
            t = os.path.join(self.tiff_dir, base + '_features.tif')
            j = os.path.join(self.json_dir, base + '.json')
            if not (os.path.exists(t) and os.path.exists(j)): continue
            is_human = (base in self.human_set)
            if not is_human:
                m = re.search(r'chip_(\d+)', base)
                if m and int(m.group(1)) <= 377: is_human = True
            self.samples.append({'base': base, 'tiff': t, 'json': j, 'is_human': is_human})

        # Quadrant expansion: each sample → 4 quadrants (0,1,2,3)
        if self.quadrant_mode:
            expanded = []
            for s in self.samples:
                for q in range(4):
                    expanded.append({**s, 'quadrant': q})
            self.samples = expanded

        n_h = sum(1 for s in self.samples if s['is_human'])
        mode = f'quadrant' if self.quadrant_mode else 'random-crop'
        print(f"Dataset: {len(self.samples)} samples ({n_h} human, {mode}, aug={augment})")

    def __len__(self): return len(self.samples)

    def _load_feat(self, p):
        with rasterio.open(p) as s:
            return s.read([i+1 for i in KEEP_BANDS]).astype(np.float32)  # rasterio 1-indexed

    def _load_label(self, p):
        with open(p, 'r', encoding='utf-8') as f: data = json.load(f)
        h, w = data['imageHeight'], data['imageWidth']
        m = np.full((h, w), -1, dtype=np.int64)
        for shape in data['shapes']:
            if shape['label'] not in LABEL_MAP: continue
            pts = shape['points']
            if len(pts) < 3: continue
            poly = Image.new('L', (w, h), 0)
            ImageDraw.Draw(poly).polygon([(p[0], p[1]) for p in pts], fill=1)
            m[np.array(poly) > 0] = LABEL_MAP[shape['label']]
        return m

    def _norm(self, f):
        for c in range(f.shape[0]):
            b = f[c]; lo, hi = b.min(), b.max()
            f[c] = (b - lo) / (hi - lo) if hi > lo else 0.0
        return f

    def _aug(self, f, m):
        if not self.augment: return f, m
        if np.random.rand() > 0.5: f, m = np.fliplr(f).copy(), np.fliplr(m).copy()
        if np.random.rand() > 0.5: f, m = np.flipud(f).copy(), np.flipud(m).copy()
        return f, m

    def _crop(self, f, m, quadrant=None):
        h, w = f.shape[1], f.shape[2]
        ps = self.patch_size
        if h <= ps: return f, m
        if quadrant is not None:
            # Fixed quadrant: 0=TL, 1=TR, 2=BL, 3=BR (for 512→256)
            rows, cols = h // ps, w // ps  # rows=2, cols=2 for 512→256
            r, c = divmod(quadrant, cols)
            y, x = r * ps, c * ps
        elif self.augment:
            # Random crop (training augmentation)
            y = np.random.randint(0, h - ps)
            x = np.random.randint(0, w - ps)
        else:
            # Center crop (evaluation)
            y = (h - ps) // 2
            x = (w - ps) // 2
        return f[:, y:y+ps, x:x+ps], m[y:y+ps, x:x+ps]

    def __getitem__(self, idx):
        s = self.samples[idx]
        f, m = self._load_feat(s['tiff']), self._load_label(s['json'])
        f, m = self._crop(f, m, quadrant=s.get('quadrant'))
        f = self._norm(f)
        f, m = self._aug(f, m)
        return torch.from_numpy(f).float(), torch.from_numpy(m).long(), s['base'], s['is_human']

def collate_fn(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            [b[2] for b in batch],
            torch.tensor([b[3] for b in batch]))
