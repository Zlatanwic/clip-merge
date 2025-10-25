import argparse
import csv
import os
from typing import Dict, Iterable, List, Optional, Tuple
from scipy.ndimage import sobel
import numpy as np
from PIL import Image
from scipy.stats import pearsonr
from skimage.measure import shannon_entropy
from tqdm import tqdm

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def read_gray(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size is not None and img.size != (size[1], size[0]):
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) 


def mutual_information(img1, img2, bins=256):
    eps = 1e-12
    hgram, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins, range=[[0, 255], [0, 255]])
    pxy = hgram / (np.sum(hgram) + eps)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = np.outer(px, py) + eps
    nzs = pxy > 0
    return float(np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs])))



def correlation_coeff(img1, img2):
    a = img1.ravel() - np.mean(img1)
    b = img2.ravel() - np.mean(img2)
    denom = (np.std(a) * np.std(b) + 1e-12)
    return float(np.mean(a * b) / denom)


def spatial_frequency(img: np.ndarray) -> float:
    rf = np.sqrt(np.mean(np.diff(img, axis=0) ** 2))
    cf = np.sqrt(np.mean(np.diff(img, axis=1) ** 2))
    return float(np.sqrt(rf**2 + cf**2))


def average_gradient(img: np.ndarray) -> float:
    gx, gy = np.gradient(img)
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


def q_abf(imgA: np.ndarray, imgB: np.ndarray, fused: np.ndarray) -> float:
    eps = 1e-8
    grad_xA, grad_yA = sobel(imgA, axis=1), sobel(imgA, axis=0)
    grad_xB, grad_yB = sobel(imgB, axis=1), sobel(imgB, axis=0)
    grad_xF, grad_yF = sobel(fused, axis=1), sobel(fused, axis=0)

    magA = np.sqrt(grad_xA**2 + grad_yA**2) + eps
    magB = np.sqrt(grad_xB**2 + grad_yB**2) + eps
    magF = np.sqrt(grad_xF**2 + grad_yF**2) + eps

    oriA = np.arctan2(grad_yA, grad_xA)
    oriB = np.arctan2(grad_yB, grad_xB)
    oriF = np.arctan2(grad_yF, grad_xF)

    qA = (2 * magA * magF + eps) / (magA**2 + magF**2 + eps)
    qB = (2 * magB * magF + eps) / (magB**2 + magF**2 + eps)
    deltaA = 1 - np.abs(np.sin(oriA - oriF))
    deltaB = 1 - np.abs(np.sin(oriB - oriF))
    qAF = qA * deltaA
    qBF = qB * deltaB

    weightA = magA / (magA + magB + eps)
    weightB = 1 - weightA

    return float(np.mean(weightA * qAF + weightB * qBF))


def entropy(img: np.ndarray) -> float:
    return float(shannon_entropy(img))


def evaluate_pair(pathA: str, pathB: str, pathF: str) -> Dict[str, float]:
    fused = read_gray(pathF)
    target_hw = (fused.shape[0], fused.shape[1])
    imgA = read_gray(pathA, size=target_hw)
    imgB = read_gray(pathB, size=target_hw)

    q_mi = mutual_information(fused, imgA) + mutual_information(fused, imgB)
    q_cb = 0.5 * (
        abs(correlation_coeff(fused, imgA)) + abs(correlation_coeff(fused, imgB))
    )
    q_sf = spatial_frequency(fused)
    en = entropy(fused)
    ag = average_gradient(fused)
    q_abf_val = q_abf(imgA, imgB, fused)

    return {
        "Q_MI": q_mi,
        "Q_CB": q_cb,
        "Q_SF": q_sf,
        "EN": en,
        "AG": ag,
        "Q_ABF": q_abf_val,
    }


def normalize_key(
    name: str,
    strip_suffix: str,
    strategy: str,
    delimiter: str,
    numeric: bool,
) -> str:
    base, _ = os.path.splitext(name)
    if strip_suffix and base.endswith(strip_suffix):
        base = base[: -len(strip_suffix)]
    if strategy == "suffix" and delimiter and delimiter in base:
        base = base.split(delimiter, 1)[1]
    if numeric:
        try:
            base = str(int(base))
        except ValueError as exc:
            raise ValueError(f"Cannot convert '{base}' to integer for file '{name}'.") from exc
    return base


def collect_mappings(
    source1: str,
    source2: str,
    fused: str,
    strategy: str,
    delimiter: str,
    fused_strip_suffix: str,
    numeric: bool,
) -> Tuple[List[str], List[Dict[str, str]]]:
    folders = [source1, source2, fused]
    mappings: List[Dict[str, str]] = []
    for idx, folder in enumerate(folders):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Directory not found: {folder}")
        strip = fused_strip_suffix if idx == 2 else ""
        mapping: Dict[str, str] = {}
        entries = [
            f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTS)
        ]
        if not entries:
            raise RuntimeError(f"No image files found in {folder}")
        for name in entries:
            key = normalize_key(name, strip, strategy, delimiter, numeric)
            if key in mapping:
                raise RuntimeError(
                    f"Duplicate key '{key}' detected in {folder}. "
                    "Adjust naming or match parameters."
                )
            mapping[key] = name
        mappings.append(mapping)

    common_keys = set.intersection(*(set(m.keys()) for m in mappings))
    if not common_keys:
        raise RuntimeError(
            "No common filenames across the provided directories after matching."
        )
    return sorted(common_keys), mappings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Q_MI / Q_CB / Q_SF / EN / AG / Q_ABF for fused images."
    )
    parser.add_argument("--source1", type=str, required=True)
    parser.add_argument("--source2", type=str, required=True)
    parser.add_argument("--fused", type=str, required=True)
    parser.add_argument(
        "--match_strategy",
        type=str,
        default="name",
        choices=["name", "suffix"],
        help="How to align filenames across directories.",
    )
    parser.add_argument(
        "--match_delimiter",
        type=str,
        default="_",
        help="Delimiter used when --match_strategy=suffix.",
    )
    parser.add_argument(
        "--match_numeric",
        action="store_true",
        help="Normalize matched keys as integers (useful when files like 001 vs 1).",
    )
    parser.add_argument(
        "--fused_strip_suffix",
        type=str,
        default="_fused",
        help="Suffix stripped from fused filenames before matching (empty string to disable).",
    )
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    keys, mappings = collect_mappings(
        args.source1,
        args.source2,
        args.fused,
        args.match_strategy,
        args.match_delimiter,
        args.fused_strip_suffix,
        args.match_numeric,
    )
    print(f"[INFO] Found {len(keys)} matched triplets.")

    results: List[Dict[str, float]] = []
    for key in tqdm(keys, desc="Evaluating"):
        fnameA = mappings[0][key]
        fnameB = mappings[1][key]
        fnameF = mappings[2][key]
        metrics = evaluate_pair(
            os.path.join(args.source1, fnameA),
            os.path.join(args.source2, fnameB),
            os.path.join(args.fused, fnameF),
        )
        metrics["filename"] = key
        results.append(metrics)

    if not results:
        print("[WARN] No overlapping samples were evaluated.")
        return

    print("\n[Aggregated Results]")
    metric_names = [k for k in results[0] if k != "filename"]
    for name in metric_names:
        mean_val = float(np.mean([r[name] for r in results]))
        print(f"{name:>7s}: {mean_val:.4f}")

    if args.output_csv:
        ensure_dir = os.path.dirname(args.output_csv)
        if ensure_dir:
            os.makedirs(ensure_dir, exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename"] + metric_names)
            writer.writeheader()
            writer.writerows(results)
        print(f"[INFO] Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()
