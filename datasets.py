import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np
import re  # 用于提取文件名中的数字编号

class MFIWHDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        """适配你的MFI-WHU训练集：.bmp格式 + 带前缀的文件名（如source1_crop_00001.bmp）
        目录文件格式：
        root_dir/
          source_1/     # 近聚焦图像：source1_crop_00001.bmp、source1_crop_00002.bmp...
          source_2/     # 远聚焦图像：source2_crop_00001.bmp、source2_crop_00002.bmp...
          full_clear/   # 真值图像：new_source2_crop_00001.bmp、new_source2_crop_00002.bmp...
        """
        self.root_dir = root_dir
        self.img_size = img_size
        
        # 训练集三文件夹路径
        self.source1_dir = os.path.join(root_dir, 'source_1')
        self.source2_dir = os.path.join(root_dir, 'source_2')
        self.full_clear_dir = os.path.join(root_dir, 'full_clear')
        
        # 验证目录存在性
        for dir_path in [self.source1_dir, self.source2_dir, self.full_clear_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"MFI-WHU训练集目录不存在: {dir_path}")
        
        # 核心：从文件名提取数字编号（如source1_crop_00001.bmp → 00001）
        def extract_num_id(filename):
            num_match = re.search(r'(\d+)\.jpg$', filename.lower())
            return num_match.group(1) if num_match else None
        
        # 获取各文件夹的“数字编号→原始文件名”映射
        def get_file_id_map(folder):
            file_id_map = {}
            for fname in os.listdir(folder):
                if fname.lower().endswith('.jpg'):
                    num_id = extract_num_id(fname)
                    if num_id:
                        file_id_map[num_id] = fname
            return file_id_map
        
        source1_map = get_file_id_map(self.source1_dir)
        source2_map = get_file_id_map(self.source2_dir)
        full_clear_map = get_file_id_map(self.full_clear_dir)
        
        # 找到共有的数字编号（匹配样本）
        common_num_ids = set(source1_map.keys()) & set(source2_map.keys()) & set(full_clear_map.keys())
        self.common_num_ids = sorted(list(common_num_ids))
        
        # 保存映射关系
        self.source1_map = source1_map
        self.source2_map = source2_map
        self.full_clear_map = full_clear_map
        
        # 校验样本数量
        if not self.common_num_ids:
            raise ValueError(
                "MFI-WHU训练集无匹配图像！请检查：\n"
                "1. 三个文件夹是否有.bmp格式图像；\n"
                "2. 图像文件名末尾是否有数字（如source1_crop_00001.bmp中的00001）；\n"
                "3. 三个文件夹是否有相同数字编号（如00001需在三个文件夹中都存在）"
            )
        print(f"✅ MFI-WHU训练集加载：共{len(self.common_num_ids)}张匹配图像（全量训练，无验证集）")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.common_num_ids)

    def __getitem__(self, idx):
        num_id = self.common_num_ids[idx]
        
        # 根据数字编号获取原始文件名
        source1_fname = self.source1_map[num_id]
        source2_fname = self.source2_map[num_id]
        full_clear_fname = self.full_clear_map[num_id]
        
        # 拼接路径
        source1_path = os.path.join(self.source1_dir, source1_fname)
        source2_path = os.path.join(self.source2_dir, source2_fname)
        full_clear_path = os.path.join(self.full_clear_dir, full_clear_fname)
        
        # 读取图像并预处理
        source1_img = Image.open(source1_path).convert('RGB')
        source2_img = Image.open(source2_path).convert('RGB')
        full_clear_img = Image.open(full_clear_path).convert('RGB')
        
        source1_img = self.transform(source1_img)
        source2_img = self.transform(source2_img)
        full_clear_img = self.transform(full_clear_img)
        
        # 数据增强：随机交换source1/source2
        if random.random() > 0.5:
            source1_img, source2_img = source2_img, source1_img
        
        return source1_img, source2_img, full_clear_img


class ABGTDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        """测试集加载类（适配A/B/GT文件夹，如A_001.bmp、GT_001.bmp）"""
        self.root_dir = root_dir
        self.img_size = img_size
        
        self.A_dir = os.path.join(root_dir, 'A')
        self.B_dir = os.path.join(root_dir, 'B')
        self.GT_dir = os.path.join(root_dir, 'GT')
        
        # 验证测试集目录
        for dir_path in [self.A_dir, self.B_dir, self.GT_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"测试集目录不存在: {dir_path}")
        
        # 提取测试集编号（如A_001.bmp → 001）
        def extract_ids(subdir, prefix):
            ids = []
            for filename in os.listdir(subdir):
                if filename.endswith('.bmp') and filename.startswith(prefix):
                    id_str = filename.replace(prefix, '').split('.')[0]
                    ids.append((id_str, filename))
            ids.sort(key=lambda x: x[0])
            return ids
        
        self.A_ids = extract_ids(self.A_dir, 'A_')
        self.B_ids = extract_ids(self.B_dir, 'B_')
        self.GT_ids = extract_ids(self.GT_dir, 'GT_')
        
        # 验证测试集数量
        if len(self.A_ids) != len(self.B_ids) != len(self.GT_ids) != 40:
            raise ValueError(
                f"测试集数量不匹配（需各40张）：\n"
                f"A:{len(self.A_ids)}张, B:{len(self.B_ids)}张, GT:{len(self.GT_ids)}张"
            )
        
        # 验证编号一致
        A_id_strs = [x[0] for x in self.A_ids]
        B_id_strs = [x[0] for x in self.B_ids]
        GT_id_strs = [x[0] for x in self.GT_ids]
        if A_id_strs != B_id_strs or A_id_strs != GT_id_strs:
            raise ValueError("测试集A/B/GT编号不匹配（如A_001需对应B_001和GT_001）")
        
        # 预处理（与训练集一致）
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.A_ids)

    def __getitem__(self, idx):
        a_id_str, a_filename = self.A_ids[idx]
        b_id_str, b_filename = self.B_ids[idx]
        gt_id_str, gt_filename = self.GT_ids[idx]
        
        if a_id_str != b_id_str or a_id_str != gt_id_str:
            raise RuntimeError(
                f"索引{idx}编号不匹配：A:{a_id_str}, B:{b_id_str}, GT:{gt_id_str}"
            )
        
        # 加载图像
        a_path = os.path.join(self.A_dir, a_filename)
        b_path = os.path.join(self.B_dir, b_filename)
        gt_path = os.path.join(self.GT_dir, gt_filename)
        
        a_img = Image.open(a_path).convert('RGB')
        b_img = Image.open(b_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        
        # 预处理
        a_img = self.transform(a_img)
        b_img = self.transform(b_img)
        gt_img = self.transform(gt_img)
        
        return a_img, b_img, gt_img


def get_mfiwh_train_loader(root_dir, batch_size=32, num_workers=4):
    """获取MFI-WHU训练集加载器（全量训练，无验证集）"""
    train_dataset = MFIWHDataset(
        root_dir=root_dir,
        img_size=512
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"📊 训练加载器配置：")
    print(f"  - 总样本数：{len(train_dataset)}张，批次数：{len(train_loader)}")
    print(f"  - 批次大小：{batch_size}，线程数：{num_workers}")
    return train_loader


def get_abgt_test_dataloader(test_root_dir, batch_size=32, num_workers=4):
    """获取测试集（A/B/GT）加载器"""
    test_dataset = ABGTDataset(
        root_dir=test_root_dir,
        img_size=512
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"\n📊 测试加载器配置：")
    print(f"  - 总样本数：{len(test_dataset)}张（固定40张），批次数：{len(test_loader)}")
    print(f"  - 批次大小：{batch_size}，线程数：{num_workers}")
    return test_loader