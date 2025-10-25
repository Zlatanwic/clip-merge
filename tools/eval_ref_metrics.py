#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from skimage.metrics import structural_similarity as ssim
from skimage.filters import gaussian
from skimage.measure import shannon_entropy

# 可选依赖，若存在会多输出 VIF / FSIM
_HAS_SEWAR = False
_HAS_PIQ = False
try:
    import sewar  # type: ignore
    _HAS_SEWAR = True
except Exception:
    _HAS_SEWAR = False

try:
    import torch
    import piq  # type: ignore
    _HAS_PIQ = True
except Exception:
    _HAS_PIQ = False


VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ---------------------------
# I/O & 工具函数
# ---------------------------

def read_rgb(path: str) -> np.ndarray:
    """读成 HxWx3, uint8"""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def to_y_channel(rgb: np.ndarray) -> np.ndarray:
    """转亮度 Y (float32 0..255)"""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def resize_like(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """把 arr resize 成 target_hw=(H,W)"""
    h, w = target_hw
    if arr.ndim == 2:
        pil = Image.fromarray(arr.astype(np.uint8), mode="L")
        pil = pil.resize((w, h), Image.BILINEAR)
        return np.asarray(pil, dtype=np.uint8)
    elif arr.ndim == 3:
        pil = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        pil = pil.resize((w, h), Image.BILINEAR)
        return np.asarray(pil, dtype=np.uint8)
    else:
        raise ValueError("resize_like: unexpected array shape")


def crop_border_np(arr: np.ndarray, border: int) -> np.ndarray:
    """裁掉四周 border 像素的边框; 如果 border=0 原样返回"""
    if border <= 0:
        return arr
    if arr.ndim == 2:
        return arr[border:-border, border:-border]
    elif arr.ndim == 3:
        return arr[border:-border, border:-border, :]
    else:
        raise ValueError("crop_border_np: unexpected array shape")


def mse_value(pred: np.ndarray, target: np.ndarray) -> float:
    diff = pred.astype(np.float32) - target.astype(np.float32)
    return float(np.mean(diff * diff))


def psnr_value(pred: np.ndarray, target: np.ndarray, data_range: float = 255.0) -> float:
    mse = mse_value(pred, target)
    if mse <= 0:
        return float("inf")
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


def ssim_value(pred: np.ndarray, target: np.ndarray, data_range: float = 255.0, multichannel: bool = False) -> float:
    """
    调 skimage.metrics.structural_similarity
    multichannel=True 时会按彩色图算，
    multichannel=False 时当灰度。
    """
    return float(ssim(
        target,  # skimage 的顺序是 (im1, im2)
        pred,
        data_range=data_range,
        multichannel=multichannel,
        gaussian_weights=True,
        use_sample_covariance=False,
    ))


def ms_ssim_gray(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 255.0,
    weights=None,
    sigmas=None,
) -> float:
    """
    简单 MS-SSIM 实现 (灰度)。多尺度逐级下采样。
    用常规 5 个尺度 + Wang2003 的权重近似。
    """
    if weights is None:
        weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float32)
    if sigmas is None:
        sigmas = [1.5, 1.5, 1.5, 1.5, 1.5]

    x = pred.astype(np.float32)
    y = target.astype(np.float32)

    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mssim_list = []

    for scale in range(len(weights)):
        mu_x = gaussian(x, sigma=sigmas[scale], mode="reflect")
        mu_y = gaussian(y, sigma=sigmas[scale], mode="reflect")

        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y

        sigma_x_sq = gaussian(x * x, sigma=sigmas[scale], mode="reflect") - mu_x_sq
        sigma_y_sq = gaussian(y * y, sigma=sigmas[scale], mode="reflect") - mu_y_sq
        sigma_xy = gaussian(x * y, sigma=sigmas[scale], mode="reflect") - mu_x_mu_y

        ssim_map = ((2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)) / (
            (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        )
        mssim_list.append(np.mean(ssim_map))

        # 除最后一层外，下采样到 1/2 分辨率
        if scale < len(weights) - 1:
            x = x[::2, ::2]
            y = y[::2, ::2]

    mssim_arr = np.clip(np.array(mssim_list, dtype=np.float64), 1e-12, 1.0)
    ms_ssim_val = np.prod(mssim_arr ** weights)
    return float(ms_ssim_val)


def corrcoef_lin(pred: np.ndarray, target: np.ndarray) -> float:
    """
    融合 vs GT 的线性相关性: Pearson with zero-mean/std normalization
    """
    a = pred.astype(np.float32).ravel()
    b = target.astype(np.float32).ravel()
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = (np.std(a) * np.std(b) + 1e-12)
    return float(np.mean(a * b) / denom)


def mutual_information(img1: np.ndarray, img2: np.ndarray, bins: int = 256) -> float:
    """
    基于直方图的互信息, 0..255 范围固定
    """
    eps = 1e-12
    hgram, _, _ = np.histogram2d(
        img1.ravel(), img2.ravel(),
        bins=bins,
        range=[[0, 255], [0, 255]],
    )
    pxy = hgram / (np.sum(hgram) + eps)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = np.outer(px, py) + eps
    nzs = pxy > 0
    return float(np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs])))


def nmi_pair(pred: np.ndarray, target: np.ndarray) -> float:
    """
    归一化互信息:
    NMI = 2 * MI / (H(X)+H(Y))
    H 使用 Shannon 熵 (skimage.measure.shannon_entropy)
    """
    mi = mutual_information(pred, target)
    hx = shannon_entropy(pred)
    hy = shannon_entropy(target)
    return float((2.0 * mi) / (hx + hy + 1e-12))


# ---------------------------
# 文件名匹配逻辑
# ---------------------------

def normalize_key(filename: str, strategy: str, fused_strip_suffix: str,
                  delimiter: str, numeric: bool, is_fused: bool) -> str:
    """
    strategy='name'      -> 用完整文件名(不带扩展)
    strategy='suffix'    -> 以 delimiter 之后的部分为 key
    fused_strip_suffix   -> 比如 '_fused'，从融合图文件名末尾剥掉
    numeric=True         -> '001'/'1' 归一到 int
    """
    base, _ = os.path.splitext(filename)

    # 去掉融合结果里自带后缀，比如 xxx_fused -> xxx
    if is_fused and fused_strip_suffix and base.endswith(fused_strip_suffix):
        base = base[: -len(fused_strip_suffix)]

    if strategy == "suffix":
        if delimiter and delimiter in base:
            base = base.split(delimiter, 1)[1]

    if numeric:
        # 如果是 0001 vs 1，这里会把它们都当作 int 再转回 str
        try:
            base = str(int(base))
        except ValueError as e:
            raise ValueError(
                f"Cannot convert '{base}' to int for file '{filename}'."
            ) from e

    return base


def collect_pairs(
    gt_dir: str,
    fused_dir: str,
    strategy: str,
    delimiter: str,
    fused_strip_suffix: str,
    numeric: bool,
) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    返回:
      keys: 匹配到的样本 key 列表
      map_gt[key] = gt_filename
      map_fused[key] = fused_filename
    """
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    if not os.path.isdir(fused_dir):
        raise FileNotFoundError(f"Fused directory not found: {fused_dir}")

    gt_map: Dict[str, str] = {}
    fused_map: Dict[str, str] = {}

    gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith(VALID_EXTS)]
    if not gt_files:
        raise RuntimeError(f"No image files found in {gt_dir}")

    fused_files = [f for f in os.listdir(fused_dir) if f.lower().endswith(VALID_EXTS)]
    if not fused_files:
        raise RuntimeError(f"No image files found in {fused_dir}")

    for name in gt_files:
        key = normalize_key(
            name,
            strategy=strategy,
            fused_strip_suffix="",
            delimiter=delimiter,
            numeric=numeric,
            is_fused=False,
        )
        if key in gt_map:
            raise RuntimeError(f"Duplicate GT key {key} in {gt_dir}")
        gt_map[key] = name

    for name in fused_files:
        key = normalize_key(
            name,
            strategy=strategy,
            fused_strip_suffix=fused_strip_suffix,
            delimiter=delimiter,
            numeric=numeric,
            is_fused=True,
        )
        if key in fused_map:
            raise RuntimeError(f"Duplicate fused key {key} in {fused_dir}")
        fused_map[key] = name

    common_keys = sorted(set(gt_map.keys()) & set(fused_map.keys()))
    if not common_keys:
        raise RuntimeError("No common filenames between GT and fused dirs (after matching).")

    return common_keys, gt_map, fused_map


# ---------------------------
# 单张图评估
# ---------------------------

def evaluate_pair(
    gt_rgb: np.ndarray,
    fu_rgb: np.ndarray,
    eval_mode: str,
) -> Dict[str, float]:
    """
    eval_mode:
        'y'   -> 用亮度通道(Y)算指标
        'rgb' -> 3个通道分别算，再取平均
    """
    out: Dict[str, float] = {}

    if eval_mode == "y":
        gt_y = to_y_channel(gt_rgb)
        fu_y = to_y_channel(fu_rgb)

        out["PSNR"] = psnr_value(fu_y, gt_y, 255.0)
        out["SSIM"] = ssim_value(fu_y, gt_y, 255.0, multichannel=False)
        out["MS_SSIM"] = ms_ssim_gray(fu_y, gt_y, 255.0)
        out["MSE"] = mse_value(fu_y, gt_y)
        out["CC"] = corrcoef_lin(fu_y, gt_y)
        out["NMI"] = nmi_pair(fu_y, gt_y)

        # 可选指标
        if _HAS_SEWAR:
            try:
                out["VIF"] = float(sewar.full_ref.vifp(gt_y.astype(np.float64), fu_y.astype(np.float64)))
            except Exception:
                pass
        if _HAS_PIQ:
            try:
                t_gt = torch.from_numpy(gt_y / 255.0).unsqueeze(0).unsqueeze(0).float()
                t_fu = torch.from_numpy(fu_y / 255.0).unsqueeze(0).unsqueeze(0).float()
                out["FSIM"] = float(piq.fsim(t_fu, t_gt).item())
                out["VIF_piq"] = float(piq.vif(t_fu, t_gt).item())
            except Exception:
                pass

    elif eval_mode == "rgb":
        psnr_list, ssim_list, ms_list, mse_list, cc_list, nmi_list = [], [], [], [], [], []

        for c in range(3):
            gt_c = gt_rgb[..., c].astype(np.float32)
            fu_c = fu_rgb[..., c].astype(np.float32)

            psnr_list.append(psnr_value(fu_c, gt_c, 255.0))
            ssim_list.append(ssim_value(fu_c, gt_c, 255.0, multichannel=False))
            ms_list.append(ms_ssim_gray(fu_c, gt_c, 255.0))
            mse_list.append(mse_value(fu_c, gt_c))
            cc_list.append(corrcoef_lin(fu_c, gt_c))
            nmi_list.append(nmi_pair(fu_c, gt_c))

        out["PSNR"] = float(np.mean(psnr_list))
        out["SSIM"] = float(np.mean(ssim_list))
        out["MS_SSIM"] = float(np.mean(ms_list))
        out["MSE"] = float(np.mean(mse_list))
        out["CC"] = float(np.mean(cc_list))
        out["NMI"] = float(np.mean(nmi_list))

        if _HAS_SEWAR:
            try:
                out["VIF"] = float(sewar.full_ref.vifp(gt_rgb.astype(np.float64), fu_rgb.astype(np.float64)))
            except Exception:
                pass
        if _HAS_PIQ:
            try:
                t_gt = torch.from_numpy(gt_rgb / 255.0).permute(2,0,1).unsqueeze(0).float()
                t_fu = torch.from_numpy(fu_rgb / 255.0).permute(2,0,1).unsqueeze(0).float()
                out["FSIM"] = float(piq.fsim(t_fu, t_gt).item())
                out["VIF_piq"] = float(piq.vif(t_fu, t_gt).item())
            except Exception:
                pass
    else:
        raise ValueError("eval_mode must be 'y' or 'rgb'")

    return out


# ---------------------------
# 主流程
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch reference-based evaluation for multi-focus fusion: PSNR / SSIM / MS-SSIM / MSE / CC / NMI (+optional VIF/FSIM)."
    )

    parser.add_argument("--gt", type=str, required=True,
                        help="Directory with ground-truth all-in-focus images.")
    parser.add_argument("--fused", type=str, required=True,
                        help="Directory with fused results (from your model).")

    # 匹配规则
    parser.add_argument("--match_strategy", type=str, default="name",
                        choices=["name", "suffix"],
                        help="How to match filenames between GT and fused.")
    parser.add_argument("--match_delimiter", type=str, default="_",
                        help="Used when --match_strategy=suffix.")
    parser.add_argument("--fused_strip_suffix", type=str, default="_fused",
                        help="Strip this suffix from fused names before matching (e.g. abc_fused.png -> abc).")
    parser.add_argument("--match_numeric", action="store_true",
                        help="Normalize keys as int (so 001 and 1 match).")

    # 尺寸 + 裁边
    parser.add_argument("--resize_policy", type=str, default="fused_to_gt",
                        choices=["fused_to_gt", "gt_to_fused", "none"],
                        help="If sizes differ: resize fused to GT, or GT to fused, or 'none' (error).")
    parser.add_argument("--crop_border", type=int, default=0,
                        help="Crop N pixels off each border of BOTH GT and fused before computing metrics (to ignore black edges).")

    # 评估通道
    parser.add_argument("--eval_mode", type=str, default="y",
                        choices=["y", "rgb"],
                        help="'y' = evaluate on luminance channel only (recommended). 'rgb' = average over R/G/B channels.")

    # CSV 输出
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional CSV path to save per-image metrics.")

    args = parser.parse_args()

    # 1) 对齐文件对
    keys, map_gt, map_fu = collect_pairs(
        gt_dir=args.gt,
        fused_dir=args.fused,
        strategy=args.match_strategy,
        delimiter=args.match_delimiter,
        fused_strip_suffix=args.fused_strip_suffix,
        numeric=args.match_numeric,
    )
    print(f"[INFO] Found {len(keys)} matched pairs")

    results: List[Dict[str, float]] = []
    metric_names: Optional[List[str]] = None

    # 2) 遍历并算指标
    for k in tqdm(keys, desc="Evaluating", unit="img"):
        gt_path = os.path.join(args.gt, map_gt[k])
        fu_path = os.path.join(args.fused, map_fu[k])

        gt_rgb = read_rgb(gt_path)   # HxWx3, uint8
        fu_rgb = read_rgb(fu_path)

        # 尺寸不一致时对齐
        if gt_rgb.shape != fu_rgb.shape:
            if args.resize_policy == "fused_to_gt":
                fu_rgb = resize_like(fu_rgb, (gt_rgb.shape[0], gt_rgb.shape[1]))
            elif args.resize_policy == "gt_to_fused":
                gt_rgb = resize_like(gt_rgb, (fu_rgb.shape[0], fu_rgb.shape[1]))
            else:
                raise ValueError(
                    f"Size mismatch for key {k}: "
                    f"GT {gt_rgb.shape} vs Fused {fu_rgb.shape}"
                )

        # 裁掉边界（去黑边/避免padding伪影），注意两张图都裁
        if args.crop_border > 0:
            gt_rgb = crop_border_np(gt_rgb, args.crop_border)
            fu_rgb = crop_border_np(fu_rgb, args.crop_border)

        met = evaluate_pair(gt_rgb, fu_rgb, eval_mode=args.eval_mode)
        met["filename"] = k
        results.append(met)

        if metric_names is None:
            metric_names = [m for m in met.keys() if m != "filename"]

    if not results:
        print("[WARN] Nothing evaluated.")
        return

    # 3) 打印平均
    print("\n[Aggregated Results]")
    metric_names = [m for m in results[0].keys() if m != "filename"]
    for name in metric_names:
        vals = [r[name] for r in results if name in r]
        print(f"{name:>9s}: {float(np.mean(vals)):.4f}")

    # 4) CSV 输出
    if args.output_csv:
        outdir = os.path.dirname(args.output_csv)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename"] + metric_names)
            writer.writeheader()
            for r in results:
                row = {k: r.get(k, "") for k in ["filename"] + metric_names}
                writer.writerow(row)
        print(f"[INFO] Saved per-image metrics to {args.output_csv}")

    # 附带提示
    if not _HAS_SEWAR:
        print("[NOTE] 'sewar' not found → skipping VIF/VIFP. (pip install sewar)")
    if not _HAS_PIQ:
        print("[NOTE] 'piq' not found → skipping FSIM/VIF(piq). (pip install piq)")


if __name__ == "__main__":
    main()
