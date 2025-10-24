import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modelv2 import MultiFocusFusionModel


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NORMALIZE = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference for multi-focus fusion without evaluation."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints_mfiwh_full/best_model.pth",
        help="Path to the trained checkpoint (.pth).",
    )
    parser.add_argument(
        "--source1",
        type=str,
        default="data/MFI-WHU/source_1",
        help="Directory with the first focus stack (e.g. near-focus images).",
    )
    parser.add_argument(
        "--source2",
        type=str,
        default="data/MFI-WHU/source_2",
        help="Directory with the second focus stack (e.g. far-focus images).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/infer_only",
        help="Directory to store fused outputs.",
    )
    parser.add_argument(
        "--clip_weight",
        type=float,
        default=0.2,
        help="CLIP branch weight (keep consistent with training unless intentionally changing).",
    )
    parser.add_argument(
        "--force_size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Optional spatial size (H W) to resize both inputs before fusion.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Set to 'cuda' or 'cpu'. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--save_focus_maps",
        action="store_true",
        help="If set, saves the two focus maps alongside fused images for inspection.",
    )
    parser.add_argument(
        "--match_strategy",
        type=str,
        default="name",
        choices=["name", "suffix"],
        help=(
            "How to match files between the two directories. "
            "'name' requires identical filenames; 'suffix' strips everything before the first delimiter."
        ),
    )
    parser.add_argument(
        "--match_delimiter",
        type=str,
        default="_",
        help="Delimiter used when --match_strategy=suffix to locate the shared suffix.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_key(filename: str, strategy: str, delimiter: str) -> str:
    if strategy == "name":
        return filename

    base, _ = os.path.splitext(filename)
    if delimiter and delimiter in base:
        suffix = base.split(delimiter, 1)[1]
    else:
        suffix = base
    return suffix


def collect_common_keys(
    dirs: Iterable[str], strategy: str, delimiter: str
) -> Tuple[List[str], List[Dict[str, str]]]:
    file_maps: List[Dict[str, str]] = []
    for folder in dirs:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Directory not found: {folder}")
        mapping: Dict[str, str] = {}
        entries = [
            f
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]
        if not entries:
            raise RuntimeError(f"No image files found in {folder}")
        for name in entries:
            key = normalize_key(name, strategy, delimiter)
            if key in mapping:
                raise RuntimeError(
                    f"Duplicate match key '{key}' detected in directory {folder}. "
                    "Adjust naming or choose a different --match_strategy/--match_delimiter."
                )
            mapping[key] = name
        file_maps.append(mapping)

    common_keys = set.intersection(*(set(m.keys()) for m in file_maps))
    if not common_keys:
        raise RuntimeError("No common filenames across the provided directories.")

    return sorted(common_keys), file_maps


def load_pil_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def preprocess_image(img: Image.Image, size: Optional[Tuple[int, int]]) -> torch.Tensor:
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    return NORMALIZE(img)


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    return image * std + mean


def prepare_model(ckpt_path: str, device: torch.device, clip_weight: float) -> MultiFocusFusionModel:
    model = MultiFocusFusionModel(
        block_size=32,
        overlap=4,
        clip_patch_size=32,
        clip_stride=16,
        clip_weight=clip_weight,
        train_backbone=True,
    ).to(device)

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys in checkpoint: {unexpected}")

    model.eval()
    return model


def save_focus_map(tensor: torch.Tensor, path: str) -> None:
    arr = tensor.clamp(0, 1).squeeze().cpu().numpy()
    Image.fromarray((arr * 255).astype("uint8")).save(path)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    ensure_dir(args.output_dir)

    keys, file_maps = collect_common_keys(
        [args.source1, args.source2], args.match_strategy, args.match_delimiter
    )
    print(f"[INFO] Found {len(keys)} common images.")

    model = prepare_model(args.ckpt, device, args.clip_weight)

    progress = tqdm(keys, desc="Batch inference", unit="img")

    for key in progress:
        name1 = file_maps[0][key]
        name2 = file_maps[1][key]
        path1 = os.path.join(args.source1, name1)
        path2 = os.path.join(args.source2, name2)

        raw1 = load_pil_image(path1)
        raw2 = load_pil_image(path2)

        if args.force_size is None and raw1.size != raw2.size:
            target_size = (raw1.height, raw1.width)
        else:
            target_size = tuple(args.force_size) if args.force_size is not None else None

        img1_tensor = preprocess_image(raw1, target_size).unsqueeze(0).to(device)
        img2_tensor = preprocess_image(raw2, target_size).unsqueeze(0).to(device)

        with torch.no_grad():
            fused_norm, focus_maps = model(img1_tensor, img2_tensor)

        fused_denorm = denormalize(fused_norm.squeeze(0)).clamp(0.0, 1.0).cpu()
        fused_pil = transforms.ToPILImage()(fused_denorm)
        fused_base = (
            os.path.splitext(key)[0] if args.match_strategy == "name" else key
        )
        fused_path = os.path.join(
            args.output_dir, f"{fused_base}_fused.png"
        )
        fused_pil.save(fused_path)

        if args.save_focus_maps:
            focus_dir = os.path.join(args.output_dir, "focus_maps")
            ensure_dir(focus_dir)
            w1 = focus_maps[:, 0:1].squeeze(0)
            w2 = focus_maps[:, 1:2].squeeze(0)
            base = fused_base
            save_focus_map(w1, os.path.join(focus_dir, f"{base}_source1.png"))
            save_focus_map(w2, os.path.join(focus_dir, f"{base}_source2.png"))

    print(f"[DONE] Fused images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
