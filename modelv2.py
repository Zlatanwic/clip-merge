import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from qkcu import QKCU

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 保留原模型的FeatureExtractor（未修改）
class FeatureExtractor(nn.Module):
    """ResNet50第一层特征提取器（可选冻结，修复inplace激活问题）"""
    def __init__(self, freeze=True):
        super().__init__()
        resnet50 = models.resnet50(pretrained=True)
        # 关键修复：替换原ReLU为inplace=False的版本
        self.features = nn.Sequential(
            resnet50.conv1,  # 7x7卷积，输出64通道
            resnet50.bn1,
            nn.ReLU(inplace=False),  # 替换为非原地ReLU
            resnet50.maxpool  # 3x3 maxpool，输出尺寸减半
        )
        # 冻结参数
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.features(x)  # 现在ReLU不会原地修改输入

# 保留原模型的GradientVarianceFocusHead（未修改）
class GradientVarianceFocusHead(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # 用小头产生 1 通道 logits
        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1),
        )
        self.register_buffer('sobel_x', torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('sobel_y', torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3))

    def forward(self, x):  # x: [B,C,H,W]
        grad_x = F.conv2d(x, self.sobel_x.repeat(x.size(1),1,1,1), groups=x.size(1), padding=1)
        grad_y = F.conv2d(x, self.sobel_y.repeat(x.size(1),1,1,1), groups=x.size(1), padding=1)
        grad = (grad_x.abs() + grad_y.abs()).mean(dim=1, keepdim=True)     # [B,1,H,W]

        x_mean = F.avg_pool2d(x, 3, stride=1, padding=1)
        var = ((x - x_mean)**2).mean(dim=1, keepdim=True)                  # [B,1,H,W]

        # 归一化到 [0,1]，但不再最终 sigmoid 当权重
        grad_n = torch.sigmoid(grad)
        var_n  = torch.sigmoid(var)

        stats = torch.cat([grad_n, var_n], dim=1)  # [B,2,H,W]
        logits = self.conv(stats)                  # [B,1,H,W] —— 可学习 logits

        return logits  # 注意：返回 logits，而不是 mask



# 引入你提供的新CLIP模型
class CLIPFocusClassifier(nn.Module):
    """基于CLIP的清晰/模糊二分类分类器"""
    def __init__(self, device='cuda'):
        super(CLIPFocusClassifier, self).__init__()
        self.device = device
        import clip
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clear_prompts = [
            "a clear and sharp image",
            "a focused photograph",
            "a crisp and detailed image",
            "a well-focused picture",
            "a sharp and clear photo"
        ]
        self.unclear_prompts = [
            "a blurry and unclear image",
            "an unfocused photograph",
            "a fuzzy and unclear image",
            "a blurry picture",
            "an out of focus photo"
        ]
        self.clear_text_features = self._encode_text_prompts(self.clear_prompts)
        self.unclear_text_features = self._encode_text_prompts(self.unclear_prompts)
        self.dataset_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.dataset_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        self.clip_input_size = 224
        self.patch_batch_size = 256
        self.tensor_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _encode_text_prompts(self, prompts):
        import clip
        text_tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def encode_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def classify_focus(self, image):
        if isinstance(image, torch.Tensor):
            patches = image.unsqueeze(0)
        else:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            if isinstance(image, Image.Image):
                patches = self.tensor_transform(image).unsqueeze(0)
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")
        scores, confidences = self.batch_classify_focus(patches)
        return scores[0].item(), confidences[0].item()

    def batch_classify_focus(self, patches):
        focus_scores, confidences = self._compute_focus_scores(patches)
        return focus_scores, confidences

    def _compute_focus_scores(self, patches):
        if patches.numel() == 0:
            empty = torch.empty(0, device=self.device)
            return empty, empty
        patches = patches.to(self.device)
        patches = patches * self.dataset_std + self.dataset_mean
        patches = torch.clamp(patches, 0.0, 1.0)
        focus_scores = []
        confidences = []
        for chunk in patches.split(self.patch_batch_size, dim=0):
            chunk = F.interpolate(chunk, size=self.clip_input_size, mode='bilinear', align_corners=False)
            chunk = (chunk - self.clip_mean) / self.clip_std
            with torch.no_grad():
                features = self.clip_model.encode_image(chunk)
                features = F.normalize(features, dim=-1)
            clear_similarity = features @ self.clear_text_features.t()
            unclear_similarity = features @ self.unclear_text_features.t()
            max_clear_sim, _ = clear_similarity.max(dim=1)
            max_unclear_sim, _ = unclear_similarity.max(dim=1)
            focus_score = max_clear_sim / (max_clear_sim + max_unclear_sim + 1e-8)
            confidence = torch.maximum(max_clear_sim, max_unclear_sim)
            focus_scores.append(focus_score)
            confidences.append(confidence)
        focus_scores = torch.cat(focus_scores, dim=0)
        confidences = torch.cat(confidences, dim=0)
        return focus_scores, confidences

    def get_pixel_focus_map(self, image, patch_size=32, stride=16):
        if isinstance(image, torch.Tensor):
            tensor = image
        else:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            if isinstance(image, Image.Image):
                tensor = self.tensor_transform(image)
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")
        return self._get_pixel_focus_map_tensor(tensor, patch_size, stride)

    def _get_pixel_focus_map_tensor(self, tensor, patch_size, stride):
        if tensor.dim() != 3:
            raise ValueError(f"Expected tensor shape (3, H, W), got {tensor.shape}")
        c, h, w = tensor.shape
        device = tensor.device
        tensor = tensor.unsqueeze(0)
        patches = F.unfold(tensor, kernel_size=patch_size, stride=stride)
        L = patches.shape[-1]
        if L == 0:
            raise ValueError("Patch size larger than image.")
        patches = patches.squeeze(0).transpose(0, 1).contiguous().view(L, c, patch_size, patch_size)
        focus_scores, _ = self._compute_focus_scores(patches)
        focus_scores = focus_scores.view(1, 1, L)
        focus_patches = focus_scores.repeat(1, patch_size * patch_size, 1)
        focus_map = F.fold(focus_patches, output_size=(h, w), kernel_size=patch_size, stride=stride)
        count_patches = torch.ones((1, patch_size * patch_size, L), device=device, dtype=focus_map.dtype)
        count_map = F.fold(count_patches, output_size=(h, w), kernel_size=patch_size, stride=stride)
        focus_map = focus_map / (count_map + 1e-8)
        return focus_map.squeeze(0).squeeze(0)

    def _tensor_to_pil(self, tensor):
        device = tensor.device
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = tensor.cpu().clamp(0, 1)
        tensor = (tensor * 255).byte()
        image = Image.fromarray(tensor.permute(1, 2, 0).numpy())
        return image
class CLIPPixelFusion(nn.Module):
    """基于 CLIP 的像素级焦点融合模型（输出软焦点图）"""
    def __init__(self, device='cuda', patch_size=32, stride=16):
        super(CLIPPixelFusion, self).__init__()
        self.clip_model = CLIPFocusClassifier(device=device)
        self.device = device
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, img1, img2):
        """生成与输入尺寸匹配的软焦点图，用于后续可微融合"""
        batch_size = img1.shape[0]
        focus_maps_list = []

        for b in range(batch_size):
            focus_map1 = self.clip_model.get_pixel_focus_map(img1[b], self.patch_size, self.stride)
            focus_map2 = self.clip_model.get_pixel_focus_map(img2[b], self.patch_size, self.stride)

            focus_map1_tensor = focus_map1.unsqueeze(0).unsqueeze(0).float().to(self.device)
            focus_map2_tensor = focus_map2.unsqueeze(0).unsqueeze(0).float().to(self.device)

            focus_map1_tensor = F.interpolate(
                focus_map1_tensor,
                size=(img1.shape[2], img1.shape[3]),
                mode='bilinear',
                align_corners=False
            )
            focus_map2_tensor = F.interpolate(
                focus_map2_tensor,
                size=(img2.shape[2], img2.shape[3]),
                mode='bilinear',
                align_corners=False
            )

            focus_maps = torch.cat([focus_map1_tensor, focus_map2_tensor], dim=1)
            focus_maps = torch.clamp(focus_maps, min=1e-6)
            focus_maps = focus_maps / (focus_maps.sum(dim=1, keepdim=True) + 1e-6)

            focus_maps_list.append(focus_maps)

        focus_maps = torch.cat(focus_maps_list, dim=0)

        return focus_maps


# 修改原模型的MultiFocusFusionModel，替换CLIP部分
class MultiFocusFusionModel(nn.Module):
    """多聚焦图像融合模型（使用可微软聚焦融合）"""
    def __init__(
        self,
        block_size=32,
        overlap=4,
        clip_patch_size=32,
        clip_stride=16,
        clip_weight=0.6,
        train_backbone=False,
    ):
        super().__init__()
        self.block_size = block_size
        self.overlap = overlap
        self.clip_weight = clip_weight

        self.feature_extractor = FeatureExtractor(freeze=not train_backbone)
        self.focus_head = GradientVarianceFocusHead()
        self.qkcu = QKCU()

        self.clip_fusion = CLIPPixelFusion(
            device=device,
            patch_size=clip_patch_size,
            stride=clip_stride
        )

        self.preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, img1, img2):
        """
        输入: 两张多聚焦图像 (B, 3, H, W)
        输出: 
            fused_img: (B, 3, H, W)
            focus_maps: (B, 2, H, W)
        """
        B, C, H, W = img1.shape

        # 1. 局部特征提取
        feat1 = self.feature_extractor(img1)  # (B,64,H/4,W/4)
        feat2 = self.feature_extractor(img2)

        # ❗ focus_head 现在返回 logits（单返回值）
        logit1 = self.focus_head(feat1)       # (B,1,H/4,W/4)
        logit2 = self.focus_head(feat2)       # (B,1,H/4,W/4)

        # 2. CLIP 软焦点（不可微先验）
        with torch.no_grad():
            clip_focus_maps = self.clip_fusion(img1, img2).detach()  # (B,2,H,W)
            clip_focus_maps = torch.clamp(clip_focus_maps, min=1e-6, max=1.0)

        # 3. 可微基础 logits（注意：这里是 logits，不是概率，不要 clamp/log）
        logit1_up = F.interpolate(logit1, size=(H, W), mode='bilinear', align_corners=False)
        logit2_up = F.interpolate(logit2, size=(H, W), mode='bilinear', align_corners=False)
        base_logits = torch.cat([logit1_up, logit2_up], dim=1)  # (B,2,H,W)

        # 4. CLIP 先验转成 logits 偏置 + softmax 融合
        clip_prior_logits = torch.log(clip_focus_maps)  # 先验是概率，取 log 成为 logits 偏置
        clip_gate = self.clip_weight                 # 建议先设 0.2 ~ 0.3
        combined_logits = (1.0 - clip_gate) * base_logits + clip_gate * clip_prior_logits

        final_focus_maps = torch.softmax(combined_logits, dim=1)  # (B,2,H,W)

        # 5. 融合
        fused_img = final_focus_maps[:, 0:1] * img1 + final_focus_maps[:, 1:2] * img2
        return fused_img, final_focus_maps




# 测试代码（验证输出格式兼容性）
if __name__ == "__main__":
    # 初始化模型（使用原参数+新CLIP参数）
    model = MultiFocusFusionModel(
        block_size=32, 
        overlap=4,
        clip_patch_size=32,
        clip_stride=16
    ).to(device)
    
    # 生成测试数据（与原模型测试一致）
    img1 = torch.randn(2, 3, 512, 512).to(device)
    img2 = torch.randn(2, 3, 512, 512).to(device)
    
    # 前向传播
    with torch.no_grad():
        fused_img, focus_maps = model(img1, img2)
    
    # 验证输出尺寸与原模型一致
    print(f"输入图像尺寸: {img1.shape}")
    print(f"融合图像尺寸: {fused_img.shape}")  # 应与输入一致 (2,3,512,512)
    print(f"聚焦掩码尺寸: {focus_maps.shape}")  # 应与原模型一致 (2,2,512,512)
    

