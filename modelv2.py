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
    """ResNet50第一层特征提取器（修复inplace激活问题）"""
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


# 引入你提供的新CLIP模型
class CLIPFocusClassifier(nn.Module):
    """基于CLIP的清晰/模糊二分类分类器"""
    def __init__(self, device='cuda'):
        super(CLIPFocusClassifier, self).__init__()
        
        # 加载CLIP模型
        self.device = device
        import clip  # 确保clip库可用
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # 用于分类的文本提示
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
        
        # 对文本提示编码
        self.clear_text_features = self._encode_text_prompts(self.clear_prompts)
        self.unclear_text_features = self._encode_text_prompts(self.unclear_prompts)
        
    def _encode_text_prompts(self, prompts):
        """Encode text prompts to CLIP features"""
        import clip  # 确保clip库可用
        text_tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    def encode_image(self, image):
        """将图像编码为 CLIP 特征"""
        if isinstance(image, str):
            # 加载图像
            image = Image.open(image).convert('RGB')
        
        # 图像预处理
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # 图像编码器
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features = F.normalize(image_features, dim=-1)
        
        return image_features
    
    def classify_focus(self, image):
        """
        判断图像是清晰还是模糊
        
        返回：focus_score: 0-1 之间的浮点数（0 表示模糊，1 表示清晰）
        """
        image_features = self.encode_image(image)
        
        # 计算与“清晰”和“模糊”提示文本的相似度
        clear_similarity = torch.cosine_similarity(image_features, self.clear_text_features, dim=1)
        unclear_similarity = torch.cosine_similarity(image_features, self.unclear_text_features, dim=1)
        
        # 取每个类别的最大相似度
        max_clear_sim = torch.max(clear_similarity)
        max_unclear_sim = torch.max(unclear_similarity)
        
        # 计算焦点分数（归一化到0-1之间）
        focus_score = max_clear_sim / (max_clear_sim + max_unclear_sim + 1e-8)
        
        # 计算置信度
        confidence = torch.max(max_clear_sim, max_unclear_sim)
        
        return focus_score.item(), confidence.item()
    
    def get_pixel_focus_map(self, image, patch_size=32, stride=16):
        """
        使用滑动窗口 CLIP 分类生成像素级焦点图
        """
        if isinstance(image, torch.Tensor):
            # 如果是张量，转换为 PIL 图像
            image = self._tensor_to_pil(image)
        elif isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        focus_map = np.zeros((h, w))
        count_map = np.zeros((h, w))
        
        # 遍历图像，使用滑动窗口分类每个块
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                # 提取块
                patch = image.crop((j, i, j + patch_size, i + patch_size))
                
                # 分类块
                focus_score, _ = self.classify_focus(patch)
                
                # 将焦点分数添加到焦点图
                focus_map[i:i+patch_size, j:j+patch_size] += focus_score
                count_map[i:i+patch_size, j:j+patch_size] += 1
        
        # 计算平均焦点分数
        focus_map = np.divide(focus_map, count_map, where=count_map > 0)
        
        return focus_map
    
    def _tensor_to_pil(self, tensor):
        """将张量转换为 PIL 图像"""
        device = tensor.device
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        tensor = tensor * std + mean
        
        tensor = tensor.cpu().clamp(0, 1)
        tensor = (tensor * 255).byte()
        image = Image.fromarray(tensor.permute(1, 2, 0).numpy())
        return image


# 引入你提供的CLIPPixelFusion（适配原模型输出格式）
class CLIPPixelFusion(nn.Module):
    """基于 CLIP 的像素级焦点融合模型（适配原模型输出）"""
    def __init__(self, device='cuda', patch_size=32, stride=16):
        super(CLIPPixelFusion, self).__init__()
        self.clip_model = CLIPFocusClassifier(device=device)
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        
    def forward(self, img1, img2):
        """基于像素级 CLIP 焦点检测融合两张图像"""
        batch_size = img1.shape[0]
        fused_images = []
        focus_maps_list = []  # 只保留焦点图，与原模型输出一致
        
        for b in range(batch_size):
            # 对每张输入图计算焦点图
            focus_map1 = self.clip_model.get_pixel_focus_map(img1[b], self.patch_size, self.stride)
            focus_map2 = self.clip_model.get_pixel_focus_map(img2[b], self.patch_size, self.stride)
            
            focus_map1_tensor = torch.from_numpy(focus_map1).float().to(self.device)
            focus_map2_tensor = torch.from_numpy(focus_map2).float().to(self.device)
            
            # 焦点图插值到原图大小
            focus_map1_tensor = F.interpolate(focus_map1_tensor.unsqueeze(0).unsqueeze(0), 
                                            size=(img1.shape[2], img1.shape[3]), 
                                            mode='bilinear', align_corners=False)
            focus_map2_tensor = F.interpolate(focus_map2_tensor.unsqueeze(0).unsqueeze(0), 
                                            size=(img2.shape[2], img2.shape[3]), 
                                            mode='bilinear', align_corners=False)
            
            focus_maps = torch.cat([focus_map1_tensor, focus_map2_tensor], dim=1)  # [1, 2, H, W]
            
            # 决策：每个像素选择更清晰的来源（焦点值更高的图）
            decision_map = torch.argmax(focus_maps, dim=1, keepdim=True)  # [1, 1, H, W]
            
            decision_onehot = torch.zeros_like(focus_maps)
            decision_onehot.scatter_(1, decision_map, 1)
            
            # 融合：按决策图拼接两张图的清晰区域
            fused_img = (img1[b:b+1] * decision_onehot[:, 0:1] + 
                        img2[b:b+1] * decision_onehot[:, 1:2])
            
            fused_images.append(fused_img)
            focus_maps_list.append(focus_maps)
        
        fused_img = torch.cat(fused_images, dim=0)
        focus_maps = torch.cat(focus_maps_list, dim=0)
        
        return fused_img, focus_maps  # 输出格式与原模型保持一致


# 修改原模型的MultiFocusFusionModel，替换CLIP部分
class MultiFocusFusionModel(nn.Module):
    """多聚焦图像融合模型（使用新CLIP模型）"""
    def __init__(self, block_size=32, overlap=4, clip_patch_size=32, clip_stride=16):
        super().__init__()
        self.block_size = block_size  # 保留原参数（兼容其他模块）
        self.overlap = overlap        
        # 核心模块：保留原特征提取部分
        self.feature_extractor = FeatureExtractor()
        self.focus_head = GradientVarianceFocusHead()
        self.qkcu = QKCU()
        
        # 替换原CLIPBlockClassifier为新的CLIP融合模块
        self.clip_fusion = CLIPPixelFusion(
            device=device,
            patch_size=clip_patch_size,
            stride=clip_stride
        )
        
        # 保留原图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, img1, img2):
        """
        输入: 两张多聚焦图像 (B, 3, H, W)
        输出: 
            fused_img: 融合图像 (B, 3, H, W)
            focus_maps: 聚焦掩码 (B, 2, H, W)，[:,0]对应img1权重，[:,1]对应img2权重
        """
        B, C, H, W = img1.shape
        
        # 1. 保留原特征提取与聚焦掩码（局部聚焦信息）
        # 图1处理
        feat1 = self.feature_extractor(img1)  # (B, 64, H/4, W/4)
        feat1_focus, mask1 = self.focus_head(feat1)  # mask1: (B, 1, H/4, W/4)
        
        # 图2处理
        feat2 = self.feature_extractor(img2)
        feat2_focus, mask2 = self.focus_head(feat2)  # mask2: (B, 1, H/4, W/4)
        
        # 2. 使用新CLIP模型进行融合（替换原分块分类逻辑）
        fused_img, clip_focus_maps = self.clip_fusion(img1, img2)
        
        # 3. 保留原聚焦掩码优化逻辑（融合CLIP结果与低级特征）
        mask1_up = F.interpolate(mask1, size=(H, W), mode='bilinear')
        mask2_up = F.interpolate(mask2, size=(H, W), mode='bilinear')
        
        # 融合CLIP焦点图和低级特征焦点图（保持原权重比例7:3）
        final_focus_maps = torch.cat([
            0.7 * clip_focus_maps[:, 0:1] + 0.3 * mask1_up,
            0.7 * clip_focus_maps[:, 1:2] + 0.3 * mask2_up
        ], dim=1)
        
        # 确保权重和为1
        focus_sum = final_focus_maps.sum(dim=1, keepdim=True) + 1e-8
        final_focus_maps = final_focus_maps / focus_sum
        
        return fused_img, final_focus_maps  # 输出格式与原模型完全一致


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
    