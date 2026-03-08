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
# class FeatureExtractor(nn.Module):
#     """ResNet50第一层特征提取器（可选冻结，修复inplace激活问题）"""
#     def __init__(self, freeze=True):
#         super().__init__()
#         resnet50 = models.resnet50(pretrained=True)
#         # 关键修复：替换原ReLU为inplace=False的版本
#         self.features = nn.Sequential(
#             resnet50.conv1,  # 7x7卷积，输出64通道
#             resnet50.bn1,
#             nn.ReLU(inplace=False),  # 替换为非原地ReLU
#             resnet50.maxpool  # 3x3 maxpool，输出尺寸减半
#         )
#         # 冻结参数
#         if freeze:
#             for param in self.features.parameters():
#                 param.requires_grad = False

#     def forward(self, x):
#         return self.features(x)  # 现在ReLU不会原地修改输入


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models

def window_partition(x, window_size):
    """
    将特征图划分为窗口
    Args:
        x: (B, C, H, W)
        window_size: int
    Returns:
        windows: (num_windows * B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    # 变形为 (B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    # 交换维度并合并窗口轴
    # (B, H//ws, W//ws, C, ws, ws) -> (B*nw, C, ws, ws)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    将窗口还原回特征图
    Args:
        windows: (num_windows * B, C, window_size, window_size)
        H, W: 原始特征图的高宽 (必须被 window_size 整除)
    Returns:
        x: (B, C, H, W)
    """
    # 计算 Batch size
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    C = windows.shape[1]
    
    x = windows.view(B, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return x

class ChannelCrossAttention(nn.Module):
    """
    窗口级跨模态注意力 (实际上是 Spatial Window Cross-Attention)
    优化：移除 for 循环，使用向量化计算
    """
    def __init__(self, channels, window_size=16):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1)
        self.key   = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.proj  = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5
        self.window_size = window_size
        
        # 初始化
        for m in [self.query, self.key, self.value, self.proj]:
            nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat_a, feat_b):
        B, C, H, W = feat_a.shape
        ws = self.window_size

        # 1. Padding: 确保 H, W 能被 window_size 整除
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            feat_a = F.pad(feat_a, (0, pad_w, 0, pad_h))
            feat_b = F.pad(feat_b, (0, pad_w, 0, pad_h))
        
        Hp, Wp = feat_a.shape[2], feat_a.shape[3]

        # 2. 计算 Q, K, V
        q = self.query(feat_a)
        k = self.key(feat_b)
        v = self.value(feat_b)

        # 3. 划分窗口 (Vectorized) -> [B*num_win, C, ws, ws]
        q_win = window_partition(q, ws)
        k_win = window_partition(k, ws)
        v_win = window_partition(v, ws)

        # 4. 变形为 Attention 所需格式 [B*nw, num_heads=1, C, hw] -> 简化为 [N, C, hw]
        # 这里因为没用多头，直接把 C 当做特征维度
        N, C, _, _ = q_win.shape
        q_flat = q_win.view(N, C, -1) # [N, C, ws*ws]
        k_flat = k_win.view(N, C, -1)
        v_flat = v_win.view(N, C, -1)

        # 5. 计算注意力
        # q: [N, C, L], k: [N, C, L] -> attn: [N, L, L] (Spatial Map)
        attn = (q_flat.transpose(1, 2) @ k_flat) * self.scale
        attn = attn.softmax(dim=-1)

        # 6. 聚合值
        # v: [N, C, L], attn: [N, L, L] -> out: [N, C, L]
        out_flat = v_flat @ attn.transpose(1, 2) 
        out_win = out_flat.view(N, C, ws, ws)

        # 7. 还原窗口
        out = window_reverse(out_win, ws, Hp, Wp)

        # 8. 去除 Padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
            feat_a = feat_a[:, :, :H, :W] # 还原原始输入用于残差

        # 9. Gating 和 残差
        gate = torch.sigmoid(self.proj(out))
        return feat_a + gate * out


class GroupedChannelAttention(nn.Module):
    """
    分组窗口自注意力
    优化：移除 for 循环，增加 Padding 处理
    """
    def __init__(self, channels, groups=4, window_size=16):
        super().__init__()
        assert channels % groups == 0
        self.groups = groups
        self.window_size = window_size
        self.scale = (channels // groups) ** -0.5
        
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        nn.init.xavier_uniform_(self.to_qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        # 1. Padding
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        Hp, Wp = x.shape[2], x.shape[3]

        # 2. 计算 QKV
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1) # [B, C, Hp, Wp]

        # 3. 划分窗口 -> [B*nw, C, ws, ws]
        q_win = window_partition(q, ws)
        k_win = window_partition(k, ws)
        v_win = window_partition(v, ws)

        # 4. 处理分组 (Groups)
        # [N, C, ws, ws] -> [N, Groups, C//G, ws*ws]
        N, C, _, _ = q_win.shape
        C_g = C // self.groups
        
        q_g = q_win.view(N, self.groups, C_g, -1) # [N, G, C_g, L]
        k_g = k_win.view(N, self.groups, C_g, -1)
        v_g = v_win.view(N, self.groups, C_g, -1)

        # 5. 计算注意力 (并行计算所有 Group)
        # q: [N, G, C_g, L] -> transpose -> [N, G, L, C_g]
        # matmul: [N, G, L, C_g] @ [N, G, C_g, L] -> [N, G, L, L]
        attn = (q_g.transpose(-2, -1) @ k_g) * self.scale
        attn = attn.softmax(dim=-1)

        # 6. 聚合
        # v: [N, G, C_g, L] @ attn.T: [N, G, L, L] -> [N, G, C_g, L]
        out_g = v_g @ attn.transpose(-2, -1)
        
        # 7. 还原形状
        out_win = out_g.view(N, C, ws, ws)
        out = window_reverse(out_win, ws, Hp, Wp)

        # 8. 去除 Padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
            x = x[:, :, :H, :W]

        return self.proj(out) + x

class DilatedFFN(nn.Module):
    """(保持不变)"""
    def __init__(self, channels, expand=4, dilation=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * expand, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Conv2d(channels * expand, channels, 3, padding=dilation, dilation=dilation)
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[2].weight, gain=0.01)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        return self.net(x) + x

class HybridAttentionBlock(nn.Module):
    """(保持不变)"""
    def __init__(self, channels):
        super().__init__()
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.group_attn = GroupedChannelAttention(channels)
        self.ffn = DilatedFFN(channels)

    def forward(self, x):
        local_feat = self.local(x)
        out = self.group_attn(local_feat)
        out = self.ffn(out)
        return out

class AttentionEnhancedFeatureExtractor(nn.Module):
    """(保持不变，除了实例化逻辑)"""
    def __init__(self, freeze=True, channels=64):
        super().__init__()
        resnet50 = models.resnet50(pretrained=True)

        self.stage1 = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            nn.ReLU(inplace=False),
            resnet50.maxpool,
        )

        self.hybrid_attn = HybridAttentionBlock(channels)
        
        # 建议：如果显存允许，可以为A->B和B->A使用两个独立的模块，这样参数更灵活
        # 但共用一个模块（如你原来所写）也是完全合法的 Siamese 结构
        self.cross_attn = ChannelCrossAttention(channels)

        if freeze:
            for param in self.stage1.parameters():
                param.requires_grad = False

    def forward(self, img_a, img_b):
        feat_a = self.stage1(img_a)
        feat_b = self.stage1(img_b)

        feat_a = self.hybrid_attn(feat_a)
        feat_b = self.hybrid_attn(feat_b)

        # 注意：这里共用了一套 cross_attn 权重
        feat_a_att = self.cross_attn(feat_a, feat_b)
        feat_b_att = self.cross_attn(feat_b, feat_a)

        return feat_a_att, feat_b_att




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


class SpatialGatingModule(nn.Module):
    """
    自适应空间门控模块
    功能：根据输入的特征图，动态生成 CLIP 分支的权重图。
    输入：feat1, feat2 (B, C, H, W)
    输出：gate (B, 1, H, W), 值域 [0, 1]
    """
    def __init__(self, in_channels=64, reduction=16):
        super().__init__()
        # 将两张图的特征拼接，输入通道翻倍
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // reduction, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            
            # 输出 1 通道的权重图
            nn.Conv2d(in_channels // reduction, 1, kernel_size=1),
            nn.Sigmoid()  # 确保权重在 0-1 之间
        )
        
        # 初始化：偏置设为 0，使其初始状态接近 0.5 (即 sigmoid(0))
        # 或者可以设为稍微偏向 CLIP 或 局部特征
        for m in self.gate_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feat1, feat2):
        # 拼接特征：让网络通过对比两张图的特征来决定权重
        cat_feat = torch.cat([feat1, feat2], dim=1) 
        gate = self.gate_conv(cat_feat)
        return gate




# 修改原模型的MultiFocusFusionModel，替换CLIP部分
class MultiFocusFusionModel(nn.Module):
    """多聚焦图像融合模型（集成可学习门控）"""
    def __init__(
        self,
        block_size=32,
        overlap=4,
        clip_patch_size=32,
        clip_stride=16,
        # clip_weight=0.6,  <-- 删除这个固定参数
        train_backbone=False,
        channels=64 # 对应 FeatureExtractor 的输出通道
    ):
        super().__init__()
        self.block_size = block_size
        self.overlap = overlap
        
        # 1. 特征提取器
        self.feature_extractor = AttentionEnhancedFeatureExtractor(freeze=not train_backbone, channels=channels)
        
        # 2. 局部聚焦头 (Gradient/Variance)
        self.focus_head = GradientVarianceFocusHead()
        
        # 3. CLIP 模块 (Frozen)
        self.clip_fusion = CLIPPixelFusion(
            device=device,
            patch_size=clip_patch_size,
            stride=clip_stride
        )

        # 4. [新增] 可学习的空间门控模块
        # 它将决定最终结果中有多少成分来自 CLIP
        self.gating_module = SpatialGatingModule(in_channels=channels)
        
        # 可选：保留一个可学习的全局缩放因子，增加灵活性
        self.temperature = nn.Parameter(torch.tensor(1.0)) 

    def forward(self, img1, img2):
        B, C, H, W = img1.shape

        # 1. 提取深层特征 (B, 64, H/4, W/4)
        feat1, feat2 = self.feature_extractor(img1, img2)
        
        # 2. 计算局部特征的 Logits (B, 1, H/4, W/4)
        logit1 = self.focus_head(feat1)
        logit2 = self.focus_head(feat2)

        # 3. 计算 CLIP 软焦点图 (B, 2, H, W)
        # CLIP 输出的是概率 (sum=1)，我们需要将其视为一种先验
        with torch.no_grad():
            clip_focus_maps = self.clip_fusion(img1, img2).detach()
            # 加上极小值防止 log 出错
            clip_focus_maps = torch.clamp(clip_focus_maps, min=1e-6, max=1.0)
            # 将 CLIP 概率转回 Logits 域，以便与 logit1/2 进行加权求和
            clip_prior_logits = torch.log(clip_focus_maps) 

        # 4. [关键修改] 计算可学习的门控权重
        # gate 输出为 (B, 1, H/4, W/4)，值域 [0, 1]
        gate_low_res = self.gating_module(feat1, feat2)
        
        # 将门控图上采样到原图尺寸 (B, 1, H, W)
        gate_map = F.interpolate(gate_low_res, size=(H, W), mode='bilinear', align_corners=False)
        
        # 5. 准备局部特征的 Logits
        logit1_up = F.interpolate(logit1, size=(H, W), mode='bilinear', align_corners=False)
        logit2_up = F.interpolate(logit2, size=(H, W), mode='bilinear', align_corners=False)
        base_logits = torch.cat([logit1_up, logit2_up], dim=1) # (B, 2, H, W)

        # 6. 加权融合 Logits
        # 公式: Final = (1 - Gate) * Local + Gate * CLIP
        # Gate 越大，越相信 CLIP；Gate 越小，越相信局部梯度/方差
        
        # 注意：这里 gate_map 会自动广播到 (B, 2, H, W)
        combined_logits = ((1.0 - gate_map) * base_logits + gate_map * clip_prior_logits) / self.temperature

        # 7. 生成最终 Focus Maps
        final_focus_maps = torch.softmax(combined_logits, dim=1)  # (B, 2, H, W)

        # 8. 图像融合
        fused_img = final_focus_maps[:, 0:1] * img1 + final_focus_maps[:, 1:2] * img2
        
        # 可以在训练时返回 gate_map 监控模型更倾向于信赖谁
        return fused_img, final_focus_maps , gate_map




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
    

