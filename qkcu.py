import torch
import torch.nn as nn
import torch.nn.functional as F

class QKCU(nn.Module):
    """QKCU模块"""
    def __init__(self, in_channels=64, num_scales=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales
        
        self.scale_downs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2**s, padding=1)
            for s in range(num_scales)
        ])
        
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * num_scales, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),  # 确保非原地激活
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        
        # 初始化缓存（注册为缓冲区，但不参与梯度计算）
        self.register_buffer('caches', torch.zeros(num_scales, 1, in_channels, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        scale_feats = []
        for s in range(self.num_scales):
            feat = self.scale_downs[s](x)
            scale_feats.append(feat)
        
        # 初始化缓存（若batch size不匹配，创建新张量替换，非原地）
        if self.caches.shape[1] != B:
            # 用新张量替换，而非修改原缓存
            self.caches = torch.zeros(self.num_scales, B, C, 1, 1, device=x.device)
        
        updated_feats = []
        # 关键修复：创建新缓存列表，避免原地修改self.caches
        new_caches = []
        for s in range(self.num_scales):
            curr_feat = scale_feats[s]
            cache_feat = self.caches[s]  # 使用旧缓存
            
            # 注意力计算（不变）
            Q = self.query_conv(curr_feat)
            K = self.key_conv(cache_feat)
            attn = F.cosine_similarity(Q, K, dim=1, eps=1e-8)
            attn = attn.unsqueeze(1).softmax(dim=1)
            
            V = self.value_conv(cache_feat)
            weighted_cache = attn * V
            updated = curr_feat + weighted_cache
            updated_feats.append(updated)
            
            # 计算新缓存值（非原地）
            new_cache_val = 0.7 * updated.mean(dim=(2, 3), keepdim=True).detach() + 0.3 * cache_feat.detach()
            new_caches.append(new_cache_val)  # 存入新列表
        
        # 非原地更新缓存：用新列表替换旧缓存（整个张量替换，非局部修改）
        self.caches = torch.stack(new_caches, dim=0)
        
        # 多尺度融合（不变）
        fused_scales = []
        for s in range(self.num_scales):
            fused = F.interpolate(updated_feats[s], size=(H, W), mode='bilinear', align_corners=False)
            fused_scales.append(fused)
        
        concat_feat = torch.cat(fused_scales, dim=1)
        out = self.gate_conv(concat_feat)
        
        return out