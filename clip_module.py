import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import clip
import numpy as np

class CLIPFocusClassifier(nn.Module):
    """
    基于CLIP的清晰 / 模糊二分类分类器
    """
    def __init__(self, device='cuda'):
        super(CLIPFocusClassifier, self).__init__()
        
        # 加载CLIP模型
        self.device = device
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
        
        #计算置信度
        confidence = torch.max(max_clear_sim, max_unclear_sim)
        
        return focus_score.item(), confidence.item()
    
    def get_pixel_focus_map(self, image, patch_size=32, stride=16):
        """
        使用滑动窗口 CLIP 分类生成像素级焦点图
        参数：
        image: PIL 图像或张量
        patch_size: 要分类的块大小
        stride: 滑动窗口的步长
        返回：
        focus_map: 像素级焦点分数的 numpy 数组
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

class CLIPPixelFusion(nn.Module):
    """
   基于 CLIP 的像素级焦点融合模型
    """
    def __init__(self, device='cuda', patch_size=32, stride=16):
        super(CLIPPixelFusion, self).__init__()
        self.clip_model = CLIPFocusClassifier(device=device)
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        
    def forward(self, img1, img2):
        """
        基于像素级 CLIP 焦点检测融合两张图像
        """
        batch_size = img1.shape[0]
        fused_images = []
        decision_maps = []
        focus_maps_list = []
        
        for b in range(batch_size):
            #  对每张输入图计算焦点图
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
            #  决策：每个像素选择更清晰的来源（焦点值更高的图）
            decision_map = torch.argmax(focus_maps, dim=1, keepdim=True)  # [1, 1, H, W]
            
            decision_onehot = torch.zeros_like(focus_maps)
            decision_onehot.scatter_(1, decision_map, 1)
            # 融合：按决策图拼接两张图的清晰区域
            fused_img = (img1[b:b+1] * decision_onehot[:, 0:1] + 
                        img2[b:b+1] * decision_onehot[:, 1:2])
            
            fused_images.append(fused_img)
            decision_maps.append(decision_map)
            focus_maps_list.append(focus_maps)
        
        
        fused_img = torch.cat(fused_images, dim=0)
        decision_map = torch.cat(decision_maps, dim=0)
        focus_maps = torch.cat(focus_maps_list, dim=0)
        
        return fused_img, decision_map, focus_maps

def create_clip_focus_classifier(device='cuda'):
    """创建并返回一个CLIP焦点分类器"""
    return CLIPFocusClassifier(device=device)

def create_clip_pixel_fusion_model(device='cuda', patch_size=32, stride=16):
    """创建基于CLIP的像素级融合模型"""
    return CLIPPixelFusion(device=device, patch_size=patch_size, stride=stride) 