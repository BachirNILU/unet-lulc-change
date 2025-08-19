# unet.py  — auto-detect channels, auto-upgrade to 12ch, train + calibrate + infer
import os, json, random, math
from pathlib import Path
import numpy as np
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from skimage.exposure import match_histograms
from skimage.morphology import remove_small_objects, opening, disk

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# ---------------- Config ----------------
DATA_ROOT = Path(".")
RESOLUTION = "1m"          # or "50cm"
TILES = ["T10", "T20"]
BAND_ORDER = [1, 2, 0]     # CIR [NIR,R,G] -> pseudo-RGB [R,G,NIR]
HIST_MATCH = True

PATCH_SIZE = 512
STRIDE = 256
MIN_CHANGED_PX = 32
NEG_KEEP_RATIO = 0.05
SEED = 0

BATCH_SIZE = 6
LR = 1e-3
EPOCHS = 1000
POS_WEIGHT_FALLBACK = 6.0
NUM_WORKERS = 4

WORK_DIR = Path("change_scratch") / "patches" / RESOLUTION
FX_DIR = WORK_DIR / "features"
LB_DIR = WORK_DIR / "labels"
SP_DIR = WORK_DIR / "splits"
OUT_DIR = Path("output/resunet_change")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# use tensor cores on 3090
torch.set_float32_matmul_precision("high")

# ---------------- IO helpers ----------------
def _find_file(root: Path, pats):
    for p in pats:
        h = list(root.glob(p))
        if h: return h[0]
    for p in pats:
        h = list(root.rglob(p))
        if h: return h[0]
    return None

def find_paths(tile):
    t1 = _find_file(DATA_ROOT, [f"*2018*{RESOLUTION}*{tile}*.tif", f"*2018*{tile}*{RESOLUTION}*.tif", f"*2018*{tile}*.tif"])
    t2 = _find_file(DATA_ROOT, [f"*2021*{RESOLUTION}*{tile}*.tif", f"*2021*{tile}*{RESOLUTION}*.tif", f"*2021*{tile}*.tif"])
    m  = _find_file(DATA_ROOT, [f"change_mask*{RESOLUTION}*{tile}*.tif", f"change_mask*_18_21*{RESOLUTION}*{tile}*.tif", f"change_mask*{tile}*.tif"])
    assert t1 and t2 and m, f"Missing inputs for {tile}"
    return t1, t2, m

def open_align(img_paths):
    ref = rxr.open_rasterio(img_paths[0], masked=True).squeeze()
    out = []
    for p in img_paths:
        da = rxr.open_rasterio(p, masked=True)
        if da.rio.crs != ref.rio.crs or da.rio.transform() != ref.rio.transform() or da.rio.shape != ref.rio.shape:
            rs = Resampling.nearest if "change_mask" in Path(p).name else Resampling.bilinear
            da = da.rio.reproject_match(ref, resampling=rs)
        out.append(da)
    return ref, out

def write_tif(path, arr, ref_da, nodata=None):
    da = xr.DataArray(arr, dims=("band","y","x"))
    da = da.rio.write_crs(ref_da.rio.crs, inplace=True)
    da.rio.write_transform(ref_da.rio.transform(), inplace=True)
    if nodata is not None:
        da.rio.write_nodata(nodata, inplace=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    da.rio.to_raster(path, compress="LZW")

def sliding_windows(H, W, ps, st):
    ys = list(range(0, max(1, H-ps+1), st))
    xs = list(range(0, max(1, W-ps+1), st))
    if ys[-1] != H-ps: ys.append(H-ps)
    if xs[-1] != W-ps: xs.append(W-ps)
    return [(y,x) for y in ys for x in xs]

# ---------------- Patch maker (12ch) ----------------
def make_patches():
    random.seed(SEED); np.random.seed(SEED)
    FX_DIR.mkdir(parents=True, exist_ok=True)
    LB_DIR.mkdir(parents=True, exist_ok=True)
    SP_DIR.mkdir(parents=True, exist_ok=True)

    ids = []
    for T in TILES:
        t1p, t2p, mp = find_paths(T)
        ref, (t2, t1, msk) = open_align([t2p, t1p, mp])  # ref on t2 grid

        t1a = np.array(t1)[BAND_ORDER].astype(np.float32)
        t2a = np.array(t2)[BAND_ORDER].astype(np.float32)
        if HIST_MATCH:
            for b in range(3):
                t1a[b] = match_histograms(t1a[b], t2a[b], channel_axis=None)

        lbl = (np.array(msk)[0] > 0).astype(np.uint8)

        # add explicit diffs
        d  = t2a - t1a
        ad = np.abs(d)
        X_full = np.concatenate([t1a, t2a, d, ad], axis=0)  # [12,H,W]

        H, W = lbl.shape
        coords = sliding_windows(H, W, PATCH_SIZE, STRIDE)
        random.shuffle(coords)

        pos, neg = [], []
        for (y,x) in coords:
            sub = lbl[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            (pos if sub.sum() >= MIN_CHANGED_PX else neg).append((y,x))
        neg = neg[: int(len(neg)*NEG_KEEP_RATIO)]
        picks = pos + neg
        random.shuffle(picks)

        for (y,x) in picks:
            X = X_full[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            Y = lbl[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            pid = f"{T}_{y:05d}_{x:05d}"
            write_tif(FX_DIR/f"{pid}_IMG.tif", X.astype(np.uint16), ref)
            write_tif(LB_DIR/f"{pid}_GT.tif",  Y[None,...].astype(np.uint8), ref, nodata=255)
            ids.append(pid)

    random.shuffle(ids)
    n=len(ids); n_tr=int(0.8*n); n_va=int(0.1*n)
    train_ids = ids[:n_tr]
    val_ids   = ids[n_tr:n_tr+n_va]
    test_ids  = ids[n_tr+n_va:]

    # simple oversampling of positive-heavy train patches
    def pos_ratio(pid):
        y = rxr.open_rasterio(LB_DIR / f"{pid}_GT.tif", masked=True).values[0]
        return float((y == 1).sum()) / float(y.size)

    balanced = []
    for pid in train_ids:
        r = pos_ratio(pid)
        reps = 6 if r >= 0.10 else 4 if r >= 0.05 else 2 if r >= 0.01 else 1
        balanced.extend([pid] * reps)
    random.shuffle(balanced)
    train_ids = balanced

    Path(SP_DIR/"train.txt").write_text("\n".join(train_ids))
    Path(SP_DIR/"val.txt").write_text("\n".join(val_ids))
    Path(SP_DIR/"test.txt").write_text("\n".join(test_ids))
    print(f"Patches -> train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)}")

# ---------------- Channel detection ----------------
def detect_num_channels_from_disk():
    """Return number of channels in existing *_IMG.tif patches, or None if no patches."""
    # Prefer train split, else any feature file
    split = SP_DIR/"train.txt"
    if split.exists():
        ids = [l.strip() for l in split.read_text().splitlines() if l.strip()]
        for pid in ids:
            f = FX_DIR / f"{pid}_IMG.tif"
            if f.exists():
                with rxr.open_rasterio(f, masked=True) as da:
                    return int(da.shape[0])
    # fallback: first file in folder
    hits = sorted(FX_DIR.glob("*_IMG.tif"))
    if hits:
        with rxr.open_rasterio(hits[0], masked=True) as da:
            return int(da.shape[0])
    return None

# ---------------- Dataset ----------------
class PatchDS(Dataset):
    def __init__(self, split, normalize="dataset", means=None, stds=None, aug=True):
        self.ids = [l.strip() for l in open(SP_DIR/f"{split}.txt").read().splitlines() if l.strip()]
        self.aug = aug and split=="train"
        self.normalize = normalize

        # determine channel count from one patch
        assert len(self.ids)>0, f"No ids in {split}.txt"
        sample = FX_DIR/f"{self.ids[0]}_IMG.tif"
        with rxr.open_rasterio(sample, masked=True) as da:
            self.n_ch = int(da.shape[0])

        stats_path = WORK_DIR/"stats.json"
        if normalize=="dataset":
            need_stats = True
            if stats_path.exists():
                try:
                    st = json.loads(stats_path.read_text())
                    need_stats = (len(st.get("means",[])) != self.n_ch)
                except Exception:
                    need_stats = True
            if need_stats:
                self._compute_stats(self.n_ch)
            st = json.loads(stats_path.read_text())
            self.means = np.array(st["means"], dtype=np.float32)
            self.stds  = np.array(st["stds"],  dtype=np.float32)
        else:
            self.means = np.asarray(means, np.float32); self.stds = np.asarray(stds, np.float32)

    def _compute_stats(self, n_ch, max_samples=400):
        rng = np.random.default_rng(SEED)
        picks = self.ids if len(self.ids)<=max_samples else rng.choice(self.ids, max_samples, replace=False)
        s=np.zeros(n_ch,dtype=np.float64); ss=np.zeros(n_ch,dtype=np.float64); npx=0
        for pid in picks:
            x = rxr.open_rasterio(FX_DIR/f"{pid}_IMG.tif", masked=True).values.astype(np.float32)
            valid = np.isfinite(x); x[~valid]=0
            s += x.sum(axis=(1,2)); ss += (x**2).sum(axis=(1,2)); npx += valid[0].sum()
        means = (s/(npx+1e-6)).tolist()
        stds  = np.sqrt(np.maximum(ss/(npx+1e-6)-np.array(means)**2,1e-6)).tolist()
        (WORK_DIR/"stats.json").write_text(json.dumps({"means":means,"stds":stds}, indent=2))

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        x = rxr.open_rasterio(FX_DIR/f"{pid}_IMG.tif", masked=True).values.astype(np.float32)  # [C,h,w]
        y = rxr.open_rasterio(LB_DIR/f"{pid}_GT.tif",  masked=True).values.astype(np.int64)[0]  # [h,w]

        # normalize per-dataset
        for b in range(x.shape[0]):
            x[b] = (x[b]-self.means[b])/(self.stds[b]+1e-6)

        # radiometric augs
        if self.aug and random.random() < 0.5:
            gain = 1.0 + random.uniform(-0.15, 0.15)
            bias = random.uniform(-0.10, 0.10)
            x = x * gain + bias
        if self.aug and random.random() < 0.3:
            x = x + np.random.normal(0, 0.02, size=x.shape).astype(np.float32)
        # flips + 90° rot
        if self.aug and random.random()<0.5:
            x = np.flip(x, axis=2).copy(); y = np.flip(y, axis=1).copy()
        if self.aug and random.random()<0.5:
            x = np.flip(x, axis=1).copy(); y = np.flip(y, axis=0).copy()
        if self.aug and random.random()<0.5:
            x = np.transpose(x, (0,2,1)).copy(); y = np.transpose(y, (1,0)).copy()

        return torch.from_numpy(x), torch.from_numpy(y)

# ---------------- ResUNet ----------------
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, drop=0.15):
        super().__init__()
        self.proj = nn.Identity() if c_in==c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out)
        )
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        y = self.block(x)
        y = self.drop(y)
        y = y + self.proj(x)
        return self.relu(y)

class UNet(nn.Module):
    def __init__(self, in_ch, n_classes=2, base=32, drop=0.15):
        super().__init__()
        self.enc1 = ResBlock(in_ch, base, drop)
        self.enc2 = ResBlock(base, base*2, drop)
        self.enc3 = ResBlock(base*2, base*4, drop)
        self.enc4 = ResBlock(base*4, base*8, drop)
        self.pool = nn.MaxPool2d(2)
        self.bot  = ResBlock(base*8, base*16, drop)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = ResBlock(base*16, base*8, drop)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = ResBlock(base*8, base*4, drop)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ResBlock(base*4, base*2, drop)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = ResBlock(base*2, base, drop)

        self.head = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

# ---------------- Lightning module ----------------
class LitChange(pl.LightningModule):
    def __init__(self, in_ch=12, lr=LR, pos_weight=POS_WEIGHT_FALLBACK, alpha=0.7, beta=0.3):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(in_ch=in_ch, n_classes=2, base=32, drop=0.15)
        w = min(float(pos_weight), 12.0)  # clip for stability
        self.register_buffer("cls_w", torch.tensor([1.0, w], dtype=torch.float32))
        self.ce = nn.CrossEntropyLoss(weight=self.cls_w)
        self.alpha = alpha; self.beta = beta

    def forward(self, x): return self.model(x)

    def _tversky_loss(self, logits, y, eps=1e-6):
        p1 = torch.softmax(logits, dim=1)[:, 1]
        y1 = (y==1).float()
        tp = (p1*y1).sum(dim=(1,2))
        fp = (p1*(1-y1)).sum(dim=(1,2))
        fn = ((1-p1)*y1).sum(dim=(1,2))
        t = (tp + eps) / (tp + self.alpha*fp + self.beta*fn + eps)
        return (1 - t).mean()

    def _compute_metrics(self, logits, y, stage):
        pred = torch.argmax(logits, dim=1)
        tp1 = ((pred == 1) & (y == 1)).sum().float()
        fp1 = ((pred == 1) & (y == 0)).sum().float()
        fn1 = ((pred == 0) & (y == 1)).sum().float()
        tp0 = ((pred == 0) & (y == 0)).sum().float()
        fp0 = ((pred == 0) & (y == 1)).sum().float()
        fn0 = ((pred == 1) & (y == 0)).sum().float()
        eps = 1e-6
        iou1 = tp1 / (tp1 + fp1 + fn1 + eps)
        iou0 = tp0 / (tp0 + fp0 + fn0 + eps)
        miou = 0.5 * (iou0 + iou1)
        acc  = (pred == y).float().mean()
        acc0 = tp0 / (tp0 + fn0 + eps)
        acc1 = tp1 / (tp1 + fn1 + eps)
        prec1 = tp1 / (tp1 + fp1 + eps)
        rec1  = tp1 / (tp1 + fn1 + eps)
        f1_1  = 2 * prec1 * rec1 / (prec1 + rec1 + eps)

        self.log(f"{stage}_loss", self.last_loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_iou1", iou1, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_iou0", iou0, on_epoch=True)
        self.log(f"{stage}_miou", miou, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, on_epoch=True)
        self.log(f"{stage}_acc0", acc0, on_epoch=True)
        self.log(f"{stage}_acc1", acc1, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_prec1", prec1, on_epoch=True)
        self.log(f"{stage}_rec1",  rec1,  on_epoch=True)
        self.log(f"{stage}_f1_1",  f1_1,  prog_bar=True, on_epoch=True)

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)
        self.last_loss = self.ce(logits, y) + 0.5 * self._tversky_loss(logits, y)  # CE + Tversky
        self._compute_metrics(logits, y, stage)
        return self.last_loss

    def training_step(self, batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _): return self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=6, min_lr=1e-6, verbose=True)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_f1_1"}}

# ---------------- Calibration & inference ----------------
@torch.no_grad()
def calibrate_threshold(ckpt_path, val_dl, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = LitChange.load_from_checkpoint(ckpt_path, map_location=device).eval().to(device)
    probs, gts = [], []
    for x, y in val_dl:
        x = x.to(device)
        p = torch.softmax(model(x), dim=1)[:, 1].cpu()
        probs.append(p); gts.append(y.cpu())
    probs = torch.cat(probs, 0).reshape(-1)
    gts   = torch.cat(gts, 0).reshape(-1)
    best_t, best_f1 = 0.5, -1.0
    for t in torch.linspace(0.30, 0.99, steps=36):
        pred = (probs >= t).int()
        tp = ((pred==1)&(gts==1)).sum().item()
        fp = ((pred==1)&(gts==0)).sum().item()
        fn = ((pred==0)&(gts==1)).sum().item()
        prec = tp/(tp+fp+1e-6); rec = tp/(tp+fn+1e-6)
        f1 = 2*prec*rec/(prec+rec+1e-6)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    print(f"[calibration] best F1={best_f1:.3f} at threshold={best_t:.2f}")
    return best_t

def cosine_window(ps):
    w = np.hanning(ps)
    W = np.outer(w, w)
    return (W / W.max()).astype(np.float32)

@torch.no_grad()
def infer_full_tile(ckpt_path, tile, threshold=0.5, min_blob=64,
                    device="cuda" if torch.cuda.is_available() else "cpu"):
    # load model to know in_ch
    model = LitChange.load_from_checkpoint(ckpt_path, map_location=device).eval().to(device)
    in_ch = int(model.hparams.get("in_ch", 12))

    t1p, t2p, _ = find_paths(tile)
    ref, (t2, t1) = open_align([t2p, t1p])  # ref grid on t2

    t1a = np.array(t1)[BAND_ORDER].astype(np.float32)
    t2a = np.array(t2)[BAND_ORDER].astype(np.float32)
    if HIST_MATCH:
        for b in range(3):
            t1a[b] = match_histograms(t1a[b], t2a[b], channel_axis=None)

    if in_ch == 12:
        d  = t2a - t1a
        ad = np.abs(d)
        full = np.concatenate([t1a, t2a, d, ad], axis=0)
    elif in_ch == 6:
        full = np.concatenate([t1a, t2a], axis=0)
    else:
        raise ValueError(f"Unsupported in_ch={in_ch}")

    # normalize with dataset stats
    st = json.loads((WORK_DIR/"stats.json").read_text())
    means = np.array(st["means"], np.float32); stds = np.array(st["stds"], np.float32)

    H, W = full.shape[-2], full.shape[-1]
    win = cosine_window(PATCH_SIZE)
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)

    for y in range(0, max(1, H-PATCH_SIZE+1), STRIDE):
        for x in range(0, max(1, W-PATCH_SIZE+1), STRIDE):
            X = full[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE].copy()
            for b in range(X.shape[0]):
                X[b] = (X[b] - means[b])/(stds[b]+1e-6)
            t = torch.from_numpy(X[None,...]).to(device)
            prob = torch.softmax(model(t), dim=1)[0,1].cpu().numpy()
            if prob.shape != win.shape:
                prob = torch.nn.functional.interpolate(
                    torch.from_numpy(prob)[None,None], size=win.shape, mode="bilinear", align_corners=False
                )[0,0].numpy()
            acc[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += prob * win
            cnt[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += win

    prob_full = acc / np.maximum(cnt, 1e-6)
    mask = (prob_full >= threshold).astype(np.uint8)

    if min_blob and min_blob > 0:
        m = remove_small_objects(mask.astype(bool), min_blob)
        m = opening(m, disk(1))
        mask = m.astype(np.uint8)

    out_mask = OUT_DIR / f"{tile}_{RESOLUTION}_resunet_mask.tif"
    out_prob = OUT_DIR / f"{tile}_{RESOLUTION}_resunet_prob.tif"
    write_tif(out_mask, mask[None,...], ref, nodata=255)
    write_tif(out_prob, prob_full[None,...].astype(np.float32), ref)

# ---------------- Class weights from data ----------------
def compute_pos_weight():
    split = SP_DIR/"train.txt"
    if not split.exists() or not split.read_text().strip():
        return POS_WEIGHT_FALLBACK
    ids = [l.strip() for l in split.read_text().splitlines() if l.strip()]
    tot = pos = 0
    for pid in sorted(set(ids)):
        y = rxr.open_rasterio(LB_DIR/f"{pid}_GT.tif", masked=True).values[0]
        tot += y.size
        pos += (y == 1).sum()
    pos_frac = max(1e-6, pos / tot)
    neg_frac = 1.0 - pos_frac
    w = float(neg_frac / pos_frac)
    print(f"[class weights] pos_frac={pos_frac:.4f} → raw pos_weight≈{w:.1f} (clipped later)")
    return w

# ---------------- Main ----------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Ensure patches exist; if not, build 12ch patches
    if not (SP_DIR/"train.txt").exists():
        print("Making patches (12ch)...")
        make_patches()

    # 2) If existing patches are 6ch, auto-upgrade to 12ch
    n_ch_disk = detect_num_channels_from_disk()
    if n_ch_disk is None:
        print("No patches found — building (12ch)...")
        make_patches()
        n_ch_disk = 12
    elif n_ch_disk == 6:
        print("Detected 6-channel patches → rebuilding to 12-channel for better performance...")
        make_patches()
        n_ch_disk = 12
    else:
        print(f"Detected {n_ch_disk}-channel patches on disk.")

    # 3) Build dataloaders (stats recomputed if channel mismatch)
    train_ds = PatchDS("train", aug=True)
    val_ds   = PatchDS("val", aug=False)
    test_ds  = PatchDS("test", aug=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 4) Train
    pl.seed_everything(SEED)
    ckpt_dir = OUT_DIR/"checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val_f1_1",
        mode="max",
        filename="best-v3",
        save_top_k=1,
        save_last=True,
    )
    early_cb = EarlyStopping(
        monitor="val_f1_1",
        mode="max",
        patience=20,
        min_delta=0.002,
    )

    pos_w = compute_pos_weight()
    model = LitChange(in_ch=train_ds.n_ch, pos_weight=pos_w, alpha=0.7, beta=0.3)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        max_epochs=EPOCHS,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        callbacks=[ckpt_cb, early_cb, pl.callbacks.RichProgressBar()],
        default_root_dir=str(OUT_DIR),
    )

    trainer.fit(model, train_dl, val_dl)
    best = ckpt_cb.best_model_path or str(ckpt_dir/"best-v3.ckpt")
    trainer.test(model, test_dl, ckpt_path=best)

    # 5) Calibrate threshold on val and run full-tile inference + light postproc
    best_t = calibrate_threshold(best, val_dl)
    infer_full_tile(best, "T10", threshold=best_t, min_blob=64)
    infer_full_tile(best, "T20", threshold=best_t, min_blob=64)
    print("Wrote GeoTIFF masks and probabilities under", OUT_DIR)
