import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import clip
from PIL import Image
from qkcu import QKCU
import numpy as np

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"

class FeatureExtractor(nn.Module):
    """ResNet50第一层特征提取器"""
    def __init__(self, freeze=True):
        super().__init__()
        resnet50 = models.resnet50(pretrained=True)
       
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

class GradientVarianceFocusHead(nn.Module):
    """梯度-方差混合聚焦头：修正尺寸不匹配问题"""
    def __init__(self, in_channels=64):
        super().__init__()
        # 可学习参数：平衡梯度和方差的权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始值0.5，范围会自动学习
        # 1x1卷积用于特征映射（增强表达能力）
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        
        # 预定义Sobel算子（注册为缓冲区，避免每次forward重新创建）
        self.register_buffer(
            'sobel_x', 
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        self.register_buffer(
            'sobel_y', 
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        # x: (B, in_channels, H, W) - 来自特征提取器的输出
        
        # 1. 计算梯度特征（边缘锐利度）
        # 关键修复：添加padding=1，确保卷积后尺寸与输入一致
        grad_x = F.conv2d(
            x, 
            self.sobel_x.repeat(x.shape[1], 1, 1, 1),  # 扩展到输入通道数
            groups=x.shape[1], 
            padding=1  # 新增padding，解决尺寸不匹配
        )
        grad_y = F.conv2d(
            x, 
            self.sobel_y.repeat(x.shape[1], 1, 1, 1), 
            groups=x.shape[1], 
            padding=1  # 新增padding，解决尺寸不匹配
        )
        # 计算梯度强度并在通道维度平均
        grad = torch.mean(torch.abs(grad_x) + torch.abs(grad_y), dim=1, keepdim=True)  # (B, 1, H, W)
        
        # 2. 计算方差特征（纹理丰富度）
        x_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)  # 保持尺寸
        x_var = F.avg_pool2d((x - x_mean) **2, kernel_size=3, stride=1, padding=1)  # 保持尺寸
        var = torch.mean(x_var, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # 3. 生成聚焦掩码（确保grad_norm和var_norm尺寸相同）
        grad_norm = F.sigmoid(grad)  # 归一化到[0,1]
        var_norm = F.sigmoid(var)    # 归一化到[0,1]
        
        # 融合梯度和方差信息（现在尺寸匹配，可以安全相加）
        mask = self.alpha * grad_norm + (1 - self.alpha) * var_norm
        mask = F.sigmoid(mask)  # 最终掩码，强化区分度
        
        # 4. 特征加权（增强清晰区域特征）
        x_focus = x * mask.repeat(1, x.shape[1], 1, 1)  # 扩展掩码到输入通道数
        
        return x_focus, mask


class CLIPBlockClassifier(nn.Module):
    """CLIP分块分类器（冻结预训练权重）"""
    def __init__(self, clip_model_name="ViT-B/32"):
        super().__init__()
        # 加载预训练CLIP
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        # 冻结CLIP参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 文本提示："清晰"和"不清晰"
        self.texts = clip.tokenize(["clear image block", "blurry image block"]).to(device)
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(self.texts)  # (2, 512)
        
        # 轻量分类头（可选，用于特征拼接方案）
        self.classifier = nn.Linear(512 + 64, 2)  # CLIP特征(512) + 增强特征(64)

    def forward(self, image_blocks, enhance_features):
        # image_blocks: (B, N, 3, 224, 224) - N个分块
        # enhance_features: (B, N, 64) - 对应分块的增强特征（来自QKCU）
        
        B, N = image_blocks.shape[0], image_blocks.shape[1]
        # 1. CLIP图像特征提取
        clip_features = []
        for i in range(N):
            block = image_blocks[:, i]  # (B, 3, 224, 224)
            with torch.no_grad():
                feat = self.clip_model.encode_image(block)  # (B, 512)
            clip_features.append(feat)
        clip_features = torch.stack(clip_features, dim=1)  # (B, N, 512)
        
        # 2. 特征拼接：将CLIP特征与QKCU增强特征在通道维度拼接
        concat_feat = torch.cat([clip_features, enhance_features], dim=2)  # (B, N, 512+64)
        logits = self.classifier(concat_feat)  # (B, N, 2)
        
        # 3. 计算置信度（softmax概率）
        probs = F.softmax(logits, dim=2)  # (B, N, 2)
        labels = torch.argmax(probs, dim=2)  # (B, N)：0=清晰，1=不清晰
        confidences = torch.max(probs, dim=2).values  # (B, N)：最大概率即置信度
        
        return labels, confidences


class MultiFocusFusionModel(nn.Module):
    """多聚焦图像融合模型（输出融合图像和聚焦掩码）"""
    def __init__(self, block_size=32, overlap=4):
        super().__init__()
        self.block_size = block_size  # 分块大小
        self.overlap = overlap        # 滑动窗口重叠度
        # 核心模块
        self.feature_extractor = FeatureExtractor()
        self.focus_head = GradientVarianceFocusHead()
        self.qkcu = QKCU()
        self.clip_classifier = CLIPBlockClassifier()
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def sliding_window_blocks(self, x):
        """将图像分块（滑动窗口）"""
        B, C, H, W = x.shape
        step = self.block_size - self.overlap
        num_blocks_h = (H - self.overlap) // step
        num_blocks_w = (W - self.overlap) // step
        blocks = []
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                h_start = i * step
                h_end = h_start + self.block_size
                w_start = j * step
                w_end = w_start + self.block_size
                block = x[:, :, h_start:h_end, w_start:w_end]
                block = F.interpolate(block, size=(224, 224), mode='bilinear')
                blocks.append(block)
        return torch.stack(blocks, dim=1), (num_blocks_h, num_blocks_w)

    def forward(self, img1, img2):
        """
        输入: 两张多聚焦图像 (B, 3, H, W)
        输出: 
            fused_img: 融合图像 (B, 3, H, W)
            focus_maps: 聚焦掩码 (B, 2, H, W)，[:,0]对应img1权重，[:,1]对应img2权重
        """
        B, C, H, W = img1.shape
        step = self.block_size - self.overlap
        
        # 1. 特征提取与聚焦掩码（局部聚焦信息）
        # 图1处理
        feat1 = self.feature_extractor(img1)  # (B, 64, H/4, W/4)
        feat1_focus, mask1 = self.focus_head(feat1)  # mask1: (B, 1, H/4, W/4)
        feat1_qkcu = self.qkcu(feat1_focus)
        
        # 图2处理
        feat2 = self.feature_extractor(img2)
        feat2_focus, mask2 = self.focus_head(feat2)  # mask2: (B, 1, H/4, W/4)
        feat2_qkcu = self.qkcu(feat2_focus)
        
        # 2. 分块与CLIP分类（全局决策信息）
        blocks1, (nh, nw) = self.sliding_window_blocks(img1)
        blocks2, _ = self.sliding_window_blocks(img2)
        N = blocks1.shape[1]
        
        feat1_pool = F.adaptive_avg_pool2d(feat1_qkcu, (nh, nw)).view(-1, nh*nw, 64)
        feat2_pool = F.adaptive_avg_pool2d(feat2_qkcu, (nh, nw)).view(-1, nh*nw, 64)
        
        labels1, conf1 = self.clip_classifier(blocks1, feat1_pool)  # (B, N), (B, N)
        labels2, conf2 = self.clip_classifier(blocks2, feat2_pool)
        
        # 3. 初始化融合结果和聚焦掩码（使用.clone()确保梯度可追踪）
        fused_img = torch.zeros_like(img1).clone()
        count = torch.zeros_like(img1).clone()
        focus_maps = torch.zeros(B, 2, H, W, device=img1.device).clone()
        
        # 4. 滑动窗口融合与聚焦掩码构建（消除原地操作）
        idx = 0
        for i in range(nh):
            for j in range(nw):
                h_start = i * step
                h_end = min(h_start + self.block_size, H)
                w_start = j * step
                w_end = min(w_start + self.block_size, W)
                
                # 提取当前块区域（用于非原地更新）
                region = (slice(b, b+1) for b in [h_start, h_end, w_start, w_end])
                h_slice, w_slice = slice(h_start, h_end), slice(w_start, w_end)
                
                for b in range(B):
                    # 计算权重
                    is_clear1 = (labels1[b, idx] == 0)
                    is_clear2 = (labels2[b, idx] == 0)
                    
                    if is_clear1 and not is_clear2:
    # img1块清晰，img2块模糊 → 全用img1
                            w1, w2 = 1.0, 0.0
                    elif is_clear2 and not is_clear1:
                            # img2块清晰，img1块模糊 → 全用img2
                            w1, w2 = 0.0, 1.0
                    elif is_clear1 and is_clear2:
                        # 两者都清晰 → 选择置信度更高的块（置信度高表示模型对“清晰”的判断更确信）
                        if conf1[b, idx] >= conf2[b, idx]:
                                # img1的清晰置信度更高 → 全用img1
                                w1, w2 = 1.0, 0.0
                        else:
                                # img2的清晰置信度更高 → 全用img2
                                w1, w2 = 0.0, 1.0
                    else:
                            # 两者都模糊 → 选择置信度更低的块（置信度低表示模型对“模糊”的判断更确信，规避它）
                        if conf1[b, idx] <= conf2[b, idx]:
                                # img1的模糊置信度更高（清晰置信度更低）→ 规避img1，用img2
                                w1, w2 = 0.0, 1.0
                        else:
                                # img2的模糊置信度更高（清晰置信度更低）→ 规避img2，用img1
                                w1, w2 = 1.0, 0.0
                    # 非原地更新融合图像（先克隆再赋值）
                    current_fused = fused_img[b, :, h_slice, w_slice].clone()
                    new_fused = current_fused + w1 * img1[b, :, h_slice, w_slice] + w2 * img2[b, :, h_slice, w_slice]
                    fused_img[b, :, h_slice, w_slice] = new_fused
                    
                    # 非原地更新计数
                    current_count = count[b, :, h_slice, w_slice].clone()
                    count[b, :, h_slice, w_slice] = current_count + 1
                    
                    # 非原地更新聚焦掩码（使用torch.full_like创建新张量）
                    focus_maps[b, 0, h_slice, w_slice] = torch.full_like(
                        focus_maps[b, 0, h_slice, w_slice], float(w1)
                    )
                    focus_maps[b, 1, h_slice, w_slice] = torch.full_like(
                        focus_maps[b, 1, h_slice, w_slice], float(w2)
                    )
            
            idx += 1
        
        # 处理重叠区域（取平均）
        fused_img = fused_img / count.clamp(min=1e-8)
        
        # 5. 优化聚焦掩码（使用非原地操作）
        mask1_up = F.interpolate(mask1, size=(H, W), mode='bilinear')
        mask2_up = F.interpolate(mask2, size=(H, W), mode='bilinear')
        
        # 非原地更新聚焦掩码（避免+=）
        focus_maps = torch.cat([
            0.7 * focus_maps[:, 0:1] + 0.3 * mask1_up,
            0.7 * focus_maps[:, 1:2] + 0.3 * mask2_up
        ], dim=1)
        
        # 确保权重和为1（非原地操作）
        focus_sum = focus_maps.sum(dim=1, keepdim=True) + 1e-8
        focus_maps = focus_maps / focus_sum
        
        return fused_img, focus_maps

# 测试代码
if __name__ == "__main__":
    # 初始化模型
    model = MultiFocusFusionModel(block_size=32).to(device)
    
    # 生成测试数据（两张随机图像）
    img1 = torch.randn(2, 3, 512, 512).to(device)  # 批量大小2，3通道，256x256
    img2 = torch.randn(2, 3, 512, 512).to(device)
    
    # 前向传播
    with torch.no_grad():
        fused_img = model(img1, img2)
    
    print(f"输入图像尺寸: {img1.shape}")
    print(f"融合图像尺寸: {fused_img.shape}")  # 应与输入尺寸一致
