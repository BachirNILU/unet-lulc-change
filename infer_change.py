#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer binary change mask from two GeoTIFFs (t1=earlier, t2=later).

Examples:
  python infer_change.py 2018-IR_LUXEMBOURG_1m_T20.tif 2021-IR_LUXEMBOURG_1m_T20.tif \
      --ckpt best.ckpt --stats stats.json --out outputs --diff-mode auto --threshold 0.30

Notes:
- --diff-mode auto will detect the "wrapped uint16 diff" training setup from stats.json
  and reproduce it at inference for compatibility.
- --band-order expects a comma list of indices taken from the source bands (e.g. 1,2,0 for CIR [NIR,R,G]).
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from skimage.exposure import match_histograms
from skimage.morphology import remove_small_objects, opening, disk

# ---------- model (ResUNet as in training) ----------
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, drop=0.15):
        super().__init__()
        self.proj = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
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
        self.enc2 = ResBlock(base, base * 2, drop)
        self.enc3 = ResBlock(base * 2, base * 4, drop)
        self.enc4 = ResBlock(base * 4, base * 8, drop)
        self.pool = nn.MaxPool2d(2)
        self.bot = ResBlock(base * 8, base * 16, drop)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ResBlock(base * 16, base * 8, drop)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ResBlock(base * 8, base * 4, drop)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ResBlock(base * 4, base * 2, drop)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ResBlock(base * 2, base, drop)

        self.head = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bot(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

# ---------- raster helpers ----------
def open_align_to_ref(paths):
    """Open rasters and reproject/match to paths[0] (t2) grid."""
    ref = rxr.open_rasterio(paths[0], masked=True).squeeze()
    outs = []
    for p in paths:
        da = rxr.open_rasterio(p, masked=True)
        if da.rio.crs != ref.rio.crs or da.rio.transform() != ref.rio.transform() or da.rio.shape != ref.rio.shape:
            rs = Resampling.nearest if "mask" in str(p).lower() else Resampling.bilinear
            da = da.rio.reproject_match(ref, resampling=rs)
        outs.append(da)
    return ref, outs

def write_tif(path: Path, arr, ref_da, nodata=None):
    da = xr.DataArray(arr, dims=("band","y","x"))
    da = da.rio.write_crs(ref_da.rio.crs, inplace=True)
    da.rio.write_transform(ref_da.rio.transform(), inplace=True)
    if nodata is not None:
        da.rio.write_nodata(nodata, inplace=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    da.rio.to_raster(path, compress="LZW")

def cosine_window(ps: int) -> np.ndarray:
    w = np.hanning(ps)
    W = np.outer(w, w)
    return (W / (W.max() + 1e-6)).astype(np.float32)

def sliding_windows(H, W, ps, st):
    ys = list(range(0, max(1, H - ps + 1), st))
    xs = list(range(0, max(1, W - ps + 1), st))
    if ys[-1] != H - ps: ys.append(H - ps)
    if xs[-1] != W - ps: xs.append(W - ps)
    return [(y, x) for y in ys for x in xs]

# ---------- checkpoint utils ----------
def detect_in_ch_from_ckpt(ckpt: dict, fallback: int = 12) -> int:
    hp = ckpt.get("hyper_parameters") or ckpt.get("hparams") or {}
    if "in_ch" in hp:
        return int(hp["in_ch"])
    sd = ckpt.get("state_dict") or ckpt
    for k in sd.keys():
        if k.endswith("enc1.block.0.weight"):
            return int(sd[k].shape[1])
    return fallback

def build_model_and_load(ckpt_path: Path, in_ch: int) -> nn.Module:
    model = UNet(in_ch=in_ch, n_classes=2, base=32, drop=0.15)
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state.get("state_dict", state)
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("model.", "", 1) if k.startswith("model.") else k] = v
    model_keys = set(model.state_dict().keys())
    cleaned = {k: v for k, v in cleaned.items() if k in model_keys}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model

# ---------- TTA ----------
def _dihedral_apply(x, k):
    # 0..7: (rot0, rot90, rot180, rot270) x (no flip / hflip)
    r = k % 4
    f = (k // 4) % 2
    y = np.rot90(x, r, axes=(-2, -1)).copy()
    if f == 1:
        y = np.flip(y, axis=-1).copy()
    return y

def _dihedral_inverse(y, k):
    r = k % 4
    f = (k // 4) % 2
    if f == 1:
        y = np.flip(y, axis=-1).copy()
    y = np.rot90(y, -r, axes=(-2, -1)).copy()
    return y

# ---------- core inference ----------
@torch.inference_mode()
def infer(
    t1_path: Path,
    t2_path: Path,
    out_dir: Path,
    ckpt_path: Path,
    stats_path: Path = None,
    threshold: float = 0.75,
    hist_match: bool = True,
    patch_size: int = 512,
    stride: int = 256,
    band_order=(1, 2, 0),   # CIR [NIR,R,G] -> pseudo-RGB [R,G,NIR]
    min_blob: int = 64,
    open_radius: int = 1,
    save_prob: bool = True,
    diff_mode: str = "auto",  # "auto" | "wrap" | "float"
    tta: str = "none",        # "none" | "flip" | "d4"
    sweep: list = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load checkpoint and construct model
    raw_ckpt = torch.load(ckpt_path, map_location="cpu")
    in_ch = detect_in_ch_from_ckpt(raw_ckpt, fallback=12)
    model = build_model_and_load(ckpt_path, in_ch=in_ch).to(device)
    print(f"[info] Using checkpoint '{ckpt_path.name}' expecting in_ch={in_ch}")

    # read & align rasters to t2 grid
    ref, (t2, t1) = open_align_to_ref([t2_path, t1_path])

    t1a = np.array(t1).astype(np.float32)[list(band_order)]
    t2a = np.array(t2).astype(np.float32)[list(band_order)]
    if hist_match:
        for b in range(min(3, t1a.shape[0])):
            t1a[b] = match_histograms(t1a[b], t2a[b], channel_axis=None)

    # load stats (for auto diff-mode and normalization)
    if stats_path and Path(stats_path).exists():
        st = json.loads(Path(stats_path).read_text())
        means = np.array(st["means"], np.float32)
        stds  = np.array(st["stds"],  np.float32)
    else:
        means = None
        stds  = None

    # decide diff representation
    def stats_look_wrapped(m, s):
        # huge means/stds for channels 7..9 indicate uint16 wrap in training
        if m is None or s is None or len(m) < 9:
            return False
        mm = m[6:9]; ss = s[6:9]
        return (mm > 10000).any() and (ss > 10000).any()

    use_wrap = (diff_mode == "wrap") or (diff_mode == "auto" and stats_look_wrapped(means, stds))
    if diff_mode == "auto":
        print(f"[info] diff-mode auto -> {'wrap' if use_wrap else 'float'} (based on stats.json)")
    else:
        print(f"[info] diff-mode {diff_mode}")

    # build feature stack
    if in_ch == 12:
        d  = t2a - t1a
        ad = np.abs(d)
        if use_wrap:
            d  = d.astype(np.uint16).astype(np.float32)  # mimic training wrap
            ad = ad.astype(np.uint16).astype(np.float32)
        full = np.concatenate([t1a, t2a, d, ad], axis=0)
    elif in_ch == 6:
        full = np.concatenate([t1a, t2a], axis=0)
    else:
        raise ValueError(f"Unsupported in_ch={in_ch}")

    # normalization stats (dataset stats preferred; else per-tile)
    if means is not None and stds is not None and len(means) >= full.shape[0]:
        means = means[:full.shape[0]]
        stds  = stds[:full.shape[0]]
    else:
        means = np.nanmean(full.reshape(full.shape[0], -1), axis=1).astype(np.float32)
        stds  = np.nanstd (full.reshape(full.shape[0], -1), axis=1).astype(np.float32) + 1e-6

    H, W = full.shape[-2], full.shape[-1]
    win = cosine_window(patch_size)
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)

    def model_prob(tile_np):
        t = torch.from_numpy(tile_np[None, ...]).to(device)
        logits = model(t)
        return torch.softmax(logits, dim=1)[0, 1].detach().cpu().numpy()

    def model_prob_tta(tile_np):
        if tta == "none":
            return model_prob(tile_np)
        if tta == "flip":
            ks = [0, 4]  # id + hflip
        elif tta == "d4":
            ks = list(range(8))
        else:
            ks = [0]
        outs = []
        for k in ks:
            xk = _dihedral_apply(tile_np, k)
            pk = model_prob(xk)
            outs.append(_dihedral_inverse(pk, k))
        return np.mean(outs, axis=0)

    for (y, x) in sliding_windows(H, W, patch_size, stride):
        X = full[:, y:y+patch_size, x:x+patch_size].copy()
        # pad on borders so convs are valid
        if X.shape[1] != patch_size or X.shape[2] != patch_size:
            pad_y = patch_size - X.shape[1]
            pad_x = patch_size - X.shape[2]
            X = np.pad(X, ((0,0),(0,pad_y),(0,pad_x)), mode="edge")
        for b in range(X.shape[0]):
            X[b] = (X[b] - float(means[b])) / float(stds[b] + 1e-6)
        prob = model_prob_tta(X)

        h, w = min(win.shape[0], H - y), min(win.shape[1], W - x)
        acc[y:y+h, x:x+w] += prob[:h, :w] * win[:h, :w]
        cnt[y:y+h, x:x+w] += win[:h, :w]

    prob_full = acc / np.maximum(cnt, 1e-6)

    # single-threshold or sweep
    thresholds = sweep if (sweep and len(sweep) > 0) else [threshold]

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(t2_path).stem
    from pathlib import Path as _P
    written = []

    for th in thresholds:
        mask = (prob_full >= float(th)).astype(np.uint8)
        if min_blob and min_blob > 0:
            m = remove_small_objects(mask.astype(bool), min_blob)
            if open_radius and open_radius > 0:
                m = opening(m, disk(open_radius))
            mask = m.astype(np.uint8)

        mask_tif = out_dir / f"{stem}_mask_t{int(100*th):02d}.tif" if len(thresholds)>1 else out_dir / f"{stem}_mask.tif"
        mask_png = out_dir / f"{stem}_mask_t{int(100*th):02d}.png" if len(thresholds)>1 else out_dir / f"{stem}_mask.png"
        write_tif(mask_tif, mask[None, ...], ref, nodata=255)

        try:
            import imageio.v2 as imageio
            imageio.imwrite(mask_png, (mask * 255).astype(np.uint8))
        except Exception:
            import matplotlib.pyplot as plt
            plt.imsave(str(mask_png), mask, cmap="gray")

        written.append((mask_tif, mask_png))

    if save_prob:
        prob_tif = out_dir / f"{stem}_prob.tif"
        write_tif(prob_tif, prob_full[None, ...].astype(np.float32), ref)
        print(f"[ok] wrote prob map: {prob_tif}")

    for i, (mt, mp) in enumerate(written):
        tag = f" t={thresholds[i]:.2f}" if len(written)>1 else ""
        print(f"[ok] wrote{tag}:\n  {mt}\n  {mp}")

    return written

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Change mask inference (binary) from t1/t2 GeoTIFFs.")
    p.add_argument("t1", type=Path, help="Earlier image (e.g., 2018-IR_LUXEMBOURG_1m_T20.tif)")
    p.add_argument("t2", type=Path, help="Later image (e.g., 2021-IR_LUXEMBOURG_1m_T20.tif)")
    p.add_argument("--ckpt", type=Path, default=Path("best.ckpt"), help="Lightning checkpoint (.ckpt)")
    p.add_argument("--stats", type=Path, default=Path("stats.json"), help="Dataset stats JSON (optional)")
    p.add_argument("--out", type=Path, default=Path("outputs"), help="Output folder")

    p.add_argument("--threshold", type=float, default=0.75, help="Probability threshold (ignored if --sweep set)")
    p.add_argument("--sweep", type=str, default="", help="Comma-separated thresholds to sweep, e.g. 0.22,0.26,0.30")
    p.add_argument("--no-hist-match", action="store_true", help="Disable histogram matching t1→t2")
    p.add_argument("--patch-size", type=int, default=512, help="Inference patch size")
    p.add_argument("--stride", type=int, default=256, help="Stride between patches")
    p.add_argument("--min-blob", type=int, default=64, help="Remove blobs smaller than this (# pixels); 0 to disable")
    p.add_argument("--open-radius", type=int, default=1, help="Morphological opening radius (px); 0 to disable")
    p.add_argument("--no-prob", action="store_true", help="Do not save probability GeoTIFF")
    p.add_argument("--band-order", type=str, default="1,2,0", help="Band indices to form [R,G,B] from source, e.g. '1,2,0' for CIR[NIR,R,G] or '0,1,2' for RGB")
    p.add_argument("--diff-mode", choices=["auto","wrap","float"], default="auto", help="How to build difference channels")
    p.add_argument("--tta", choices=["none","flip","d4"], default="none", help="Test-time augmentation")

    args = p.parse_args()

    band_order = tuple(int(i.strip()) for i in args.band_order.split(",")) if args.band_order else (1,2,0)
    sweep = [float(s) for s in args.sweep.split(",")] if args.sweep else None

    torch.set_float32_matmul_precision("high")
    infer(
        t1_path=args.t1,
        t2_path=args.t2,
        out_dir=args.out,
        ckpt_path=args.ckpt,
        stats_path=args.stats if args.stats and args.stats.exists() else None,
        threshold=args.threshold,
        hist_match=not args.no_hist_match,
        patch_size=args.patch_size,
        stride=args.stride,
        band_order=band_order,
        min_blob=args.min_blob,
        open_radius=args.open_radius,
        save_prob=not args.no_prob,
        diff_mode=args.diff_mode,
        tta=args.tta,
        sweep=sweep,
    )

if __name__ == "__main__":
    main()
