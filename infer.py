import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# 1) 导入你自己的模型定义
from modelv2 import MultiFocusFusionModel

# 2) 设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 3) 与训练时一致的归一化（你训练里 fallback 用的是 ImageNet 默认）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return img

def preprocess(img: Image.Image, size=None):
    """
    可选 size=(H,W) 统一两图分辨率；如果你的两张图已经同尺寸，可不传。
    """
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    tensor = to_tensor(img)  # [3,H,W] normalized
    return tensor

def save_image(tensor_chw, path):
    """
    将 [3,H,W] 的 0~1 浮点张量保存为图片。
    """
    arr = (tensor_chw.clamp(0,1).permute(1,2,0).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)

@torch.no_grad()
def infer_once(
    ckpt_path: str,
    img1_path: str,
    img2_path: str,
    out_dir: str = "./outputs",
    force_size=None,          # e.g. (H, W). 若两张图尺寸不同，传一个目标尺寸
    clip_weight: float = 0.2, # 推理时可覆盖训练时的 clip 权重（可选）
    train_backbone: bool = True  # 与训练时保持一致（决定是否解冻resnet第一层；推理不影响）
):
    os.makedirs(out_dir, exist_ok=True)

    # --- Load images ---
    raw1 = load_image(img1_path)
    raw2 = load_image(img2_path)

    if force_size is None:
        # 要求两图同尺寸；若不同，自动用第一张的尺寸对齐第二张
        if raw1.size != raw2.size:
            force_size = (raw1.height, raw1.width)

    t1 = preprocess(raw1, size=force_size)
    t2 = preprocess(raw2, size=force_size)

    # --- Batch: [B,3,H,W] ---
    img1 = t1.unsqueeze(0).to(device)
    img2 = t2.unsqueeze(0).to(device)

    # --- Build model (参数需与训练一致) ---
    model = MultiFocusFusionModel(
        block_size=32,
        overlap=4,
        clip_patch_size=32,
        clip_stride=16,
        clip_weight=clip_weight,  # 推理时也能调
        train_backbone=train_backbone
    ).to(device)

    # --- Load weights ---
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        # 兼容：如果你后来用 checkpoint 字典保存
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)

    model.eval()

    # --- Inference ---
    fused, focus_maps = model(img1, img2)  # fused: [1,3,H,W], focus_maps: [1,2,H,W]

    # --- 反归一化到 [0,1] 再保存 ---
    # 因为模型输入是标准化后的图像，输出仍在“标准化空间”的加权结果；
    # 一般需要把标准化逆操作还原回 0~1 再保存可视图。
    mean = torch.tensor(IMAGENET_MEAN, device=fused.device).view(1,3,1,1)
    std  = torch.tensor(IMAGENET_STD, device=fused.device).view(1,3,1,1)
    fused_denorm = fused * std + mean   # [1,3,H,W]

    save_image(fused_denorm[0], os.path.join(out_dir, "fused.png"))

    # 同时可视化两路权重图（0~1 灰度）
    w1 = focus_maps[:,0:1]  # [1,1,H,W]
    w2 = focus_maps[:,1:1+1]
    # 直接保存为灰度
    def save_gray(w, name):
        w01 = w[0,0].clamp(0,1).cpu().numpy()
        Image.fromarray((w01*255).astype(np.uint8)).save(os.path.join(out_dir, name))

    save_gray(w1, "weight_img1.png")
    save_gray(w2, "weight_img2.png")

    print(f"[OK] Saved to {out_dir}/fused.png (and weight maps).")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="path to best_model.pth")
    p.add_argument("--img1", type=str, required=True)
    p.add_argument("--img2", type=str, required=True)
    p.add_argument("--out", type=str, default="./outputs")
    p.add_argument("--H", type=int, default=None, help="optional resize height")
    p.add_argument("--W", type=int, default=None, help="optional resize width")
    p.add_argument("--clip_weight", type=float, default=0.2)
    args = p.parse_args()

    force_size = None
    if args.H is not None and args.W is not None:
        force_size = (args.H, args.W)

    infer_once(
        ckpt_path=args.ckpt,
        img1_path=args.img1,
        img2_path=args.img2,
        out_dir=args.out,
        force_size=force_size,
        clip_weight=args.clip_weight,
        train_backbone=True
    )
