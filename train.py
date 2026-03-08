import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # 🔥 新增：用于保存热力图

# 导入数据集和模型
from datasets import get_mfiwh_train_loader
from modelv2 import MultiFocusFusionModel

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# -------------------------- 损失函数（保持原有） --------------------------
def _torch_fspecial_gauss(size, sigma, device, dtype):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    x = torch.tensor(x, dtype=dtype, device=device)
    y = torch.tensor(y, dtype=dtype, device=device)
    g = torch.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()

def SSIM_LOSS(img1, img2, size=11, sigma=1.5):
    device, dtype = img1.device, img1.dtype
    window = _torch_fspecial_gauss(size, sigma, device, dtype)
    window = window.view(1, 1, size, size).repeat(img1.size(1), 1, 1, 1)

    K1, K2, L = 0.01, 0.03, 1.0
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2

    pad = size // 2
    img1_pad = F.pad(img1, (pad, pad, pad, pad), mode='reflect')
    img2_pad = F.pad(img2, (pad, pad, pad, pad), mode='reflect')

    mu1 = F.conv2d(img1_pad, window, stride=1, groups=img1.size(1))
    mu2 = F.conv2d(img2_pad, window, stride=1, groups=img2.size(1))
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2

    sigma1_sq = F.conv2d(img1_pad * img1_pad, window, stride=1, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2_pad * img2_pad, window, stride=1, groups=img2.size(1)) - mu2_sq
    sigma12   = F.conv2d(img1_pad * img2_pad, window, stride=1, groups=img1.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return 1 - ssim_map.mean()

def L2_LOSS(diff):
    B, C, H, W = diff.shape
    l2 = torch.norm(diff, p=2, dim=(1, 2, 3)) / (C * H * W)
    return l2.mean()

def Fro_LOSS(diff):
    B, C, H, W = diff.shape
    fro = torch.sum(diff * diff, dim=(1, 2, 3)) / (C * H * W)
    return fro.mean()

# -------------------------- 辅助损失函数 --------------------------
def create_smoothness_loss(focus_maps):
    grad_x = torch.abs(focus_maps[:, :, :, 1:] - focus_maps[:, :, :, :-1])
    grad_y = torch.abs(focus_maps[:, :, 1:, :] - focus_maps[:, :, :-1, :])
    return torch.mean(grad_x) + torch.mean(grad_y)

def create_focus_accuracy_loss(focus_maps, source1, source2, gt):
    with autocast(enabled=False):
        focus_maps = focus_maps.to(torch.float32)
        source1 = source1.to(torch.float32)
        source2 = source2.to(torch.float32)

        def calculate_sharpness(img):
            gray = torch.mean(img, dim=1, keepdim=True) if img.dim() == 4 else img.unsqueeze(0)
            laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=img.device, dtype=torch.float32).view(1, 1, 3, 3)
            return torch.abs(F.conv2d(gray, laplacian, padding=1))

        sharp1 = calculate_sharpness(source1)
        sharp2 = calculate_sharpness(source2)
        denom = sharp1 + sharp2
        denom = torch.where(denom == 0, torch.full_like(denom, 1e-6), denom)
        ideal_focus1 = sharp1 / denom
        ideal_focus2 = sharp2 / denom
        loss = (F.mse_loss(focus_maps[:, 0:1], ideal_focus1) +
                F.mse_loss(focus_maps[:, 1:2], ideal_focus2))
    return loss


# -------------------------- 辅助函数（改动部分） --------------------------
def check_normalization(tensor, name, step, writer):
    with torch.no_grad():
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        writer.add_scalar(f'Normalization/{name}_mean', mean_val, step)
        writer.add_scalar(f'Normalization/{name}_std', std_val, step)
        writer.add_scalar(f'Normalization/{name}_min', min_val, step)
        writer.add_scalar(f'Normalization/{name}_max', max_val, step)
        
        if step % 10 == 0:
            print(f"[{name} 归一化检查] 均值: {mean_val:.4f}, 标准差: {std_val:.4f}, 范围: [{min_val:.4f}, {max_val:.4f}]")

# 🔥 修改：增加了 focus_maps 和 gate_map 的接收和保存逻辑
def save_images_to_folder(img1, img2, fused_img, gt_img, focus_maps, gate_map, epoch, batch_idx, mean, std, save_dir):
    """保存训练过程中的图像（含热力图）"""
    epoch_dir = os.path.join(save_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    def denormalize(tensor):
        """反归一化：恢复到[0,1]范围"""
        tensor = tensor.detach()
        tensor = tensor * std.view(3, 1, 1) + mean.view(3, 1, 1)
        return torch.clamp(tensor, 0.0, 1.0)
    
    # 仅保存 Batch 中的第一张图
    i = 0
    img1_denorm = denormalize(img1[i])
    img2_denorm = denormalize(img2[i])
    fused_denorm = denormalize(fused_img[i])
    gt_denorm = denormalize(gt_img[i])
    
    # 转换为PIL图像并保存
    img1_pil = Image.fromarray((img1_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    img2_pil = Image.fromarray((img2_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    fused_pil = Image.fromarray((fused_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    gt_pil = Image.fromarray((gt_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    
    base_name = f'batch_{batch_idx}_sample_{i}'
    img1_pil.save(os.path.join(epoch_dir, f'{base_name}_img1.jpg'), 'JPEG', quality=95)
    img2_pil.save(os.path.join(epoch_dir, f'{base_name}_img2.jpg'), 'JPEG', quality=95)
    fused_pil.save(os.path.join(epoch_dir, f'{base_name}_fused.jpg'), 'JPEG', quality=95)
    gt_pil.save(os.path.join(epoch_dir, f'{base_name}_gt.jpg'), 'JPEG', quality=95)

    # 🔥 保存 Focus Map (img1 的权重图)
    # focus_maps: [B, 2, H, W] -> 取 [0, 0, :, :]
    f_map = focus_maps[i, 0].detach().cpu().numpy()
    plt.imsave(os.path.join(epoch_dir, f'{base_name}_FocusMap.png'), f_map, cmap='jet')

    # 🔥 保存 Gate Map (CLIP 的权重图)
    # gate_map: [B, 1, H, W] -> 取 [0, 0, :, :]
    g_map = gate_map[i, 0].detach().cpu().numpy()
    plt.imsave(os.path.join(epoch_dir, f'{base_name}_GateMap.png'), g_map, cmap='magma', vmin=0, vmax=1)


def log_images_to_tensorboard(writer, img1, img2, fused_img, gt_img, gate_map, epoch, batch_idx, mean, std):
    """将图像记录到TensorBoard（实时查看训练效果）"""
    sample_idx = 0  # 取批次中第一个样本展示
    img1_sample = img1[sample_idx]
    img2_sample = img2[sample_idx]
    fused_sample = fused_img[sample_idx]
    gt_sample = gt_img[sample_idx]
    gate_sample = gate_map[sample_idx] # 🔥
    
    def denormalize(tensor):
        tensor = tensor.detach()
        tensor = tensor * std.view(3, 1, 1) + mean.view(3, 1, 1)
        return torch.clamp(tensor, 0.0, 1.0)
    
    img1_denorm = denormalize(img1_sample)
    img2_denorm = denormalize(img2_sample)
    fused_denorm = denormalize(fused_sample)
    gt_denorm = denormalize(gt_sample)
    
    # 记录到TensorBoard（使用epoch+batch_idx作为全局步长，避免覆盖）
    global_step = epoch * 1000 + batch_idx
    writer.add_image(f'Input/Image1', img1_denorm, global_step=global_step, dataformats='CHW')
    writer.add_image(f'Input/Image2', img2_denorm, global_step=global_step, dataformats='CHW')
    writer.add_image(f'Output/Fused', fused_denorm, global_step=global_step, dataformats='CHW')
    writer.add_image(f'GroundTruth/GT', gt_denorm, global_step=global_step, dataformats='CHW')
    writer.add_image(f'Internal/GateMap', gate_sample, global_step=global_step, dataformats='CHW') # 🔥


# -------------------------- 训练主函数 --------------------------
def train_integrated_model(P):
    detect_anomaly = P.get('detect_anomaly', False)
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    os.makedirs(P['save_dir'], exist_ok=True)
    
    # 创建融合图像保存目录
    image_save_dir = os.path.join(P['save_dir'], 'fusion_results')
    os.makedirs(image_save_dir, exist_ok=True)
    print(f"融合图像将保存至: {image_save_dir}")
    
    # 1. 加载数据
    print("\n[1/5] 加载MFI-WHU全量训练集...")
    train_loader = get_mfiwh_train_loader(
        root_dir=P['data_dir'],
        batch_size=P['batch_size'],
        num_workers=P['num_workers']
    )
    print(f"✅ 全量训练集加载完成：共{len(train_loader.dataset)}张样本，{len(train_loader)}个批次")
    
    # 获取归一化参数
    try:
        data_mean = torch.tensor(train_loader.dataset.mean, device=device)
        data_std = torch.tensor(train_loader.dataset.std, device=device)
    except:
        data_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        data_std = torch.tensor([0.229, 0.224, 0.225], device=device)
    
    # 2. 初始化多聚焦融合模型
    print("\n[2/5] 初始化多聚焦融合模型...")
    model = MultiFocusFusionModel(
        block_size=P['block_size'],
        overlap=P['overlap'],
        clip_patch_size=P.get('clip_patch_size', 32),
        clip_stride=P.get('clip_stride', 16),
        # 🔥 修正：移除了 clip_weight 参数，因为现在它是可学习的 gate
        train_backbone=P.get('train_backbone', False)
    ).to(P['device'])
    print(f"✅ 模型初始化完成")
    
    # 3. 配置优化器
    print("\n[3/5] 设置优化器与学习率调度...")
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=P['lr'])
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    
    amp_device_available = torch.cuda.is_available() and 'cuda' in str(P['device']).lower()
    requested_amp = P.get('use_amp', amp_device_available)
    use_amp = requested_amp and amp_device_available
    scaler = GradScaler(enabled=use_amp)
    
    # 4. 初始化日志
    log_dir = os.path.join(P['save_dir'], 'tensorboard_logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 5. 训练循环
    print("\n[4/5] 开始全量数据训练...")
    best_train_loss = float('inf')
    log_interval = 5
    save_image_interval = 1
    
    default_loss_weights = {'ssim': 2.0, 'l2': 1.0, 'fro': 0.1, 'focus_acc': 1.5, 'smooth': 0.1}
    loss_weights = {**default_loss_weights, **P.get('loss_weights', {})}
    grad_clip = P.get('grad_clip', 1.0)

    for epoch in range(P['num_epochs']):
        model.train()
        # 🔥 新增 gate_mean 指标监控
        train_metrics = {'total': 0, 'ssim': 0, 'l2': 0, 'fro': 0, 'focus_acc': 0, 'smooth': 0, 'grad_norm': 0, 'gate_mean': 0}
        
        for batch_idx, (img1, img2, gt_img) in enumerate(tqdm(
            train_loader, desc=f'Epoch {epoch+1}/{P["num_epochs"]}'
        )):
            img1, img2, gt_img = img1.to(P['device']), img2.to(P['device']), gt_img.to(P['device'])
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=use_amp):
                # 🔥 修正：前向传播现在返回 3 个值
                fused_img, focus_maps, gate_map = model(img1, img2)
                
                loss_ssim = SSIM_LOSS(fused_img, gt_img)
                loss_l2 = L2_LOSS(fused_img - gt_img)
                loss_fro = Fro_LOSS(fused_img - gt_img)
                loss_focus_acc = create_focus_accuracy_loss(focus_maps, img1, img2, gt_img)
                loss_smooth = create_smoothness_loss(focus_maps)
                
                total_loss = (
                    loss_weights['ssim'] * loss_ssim +
                    loss_weights['l2'] * loss_l2 +
                    loss_weights['fro'] * loss_fro +
                    loss_weights['focus_acc'] * loss_focus_acc +
                    loss_weights['smooth'] * loss_smooth
                )

            # 反向传播
            grad_norm_value = 0.0
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm_value = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                grad_norm_value = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            if isinstance(grad_norm_value, torch.Tensor):
                grad_norm_value = grad_norm_value.item()
            
            # 累计指标
            train_metrics['total'] += total_loss.item()
            train_metrics['ssim'] += loss_ssim.item()
            train_metrics['l2'] += loss_l2.item()
            train_metrics['fro'] += loss_fro.item()
            train_metrics['focus_acc'] += loss_focus_acc.item()
            train_metrics['smooth'] += loss_smooth.item()
            train_metrics['grad_norm'] += grad_norm_value
            # 🔥 记录 Gate 的平均值 (Detach 防止梯度泄露)
            train_metrics['gate_mean'] += gate_map.detach().mean().item()
            
            # 定期记录图像
            if batch_idx % log_interval == 0:
                log_images_to_tensorboard(
                    writer=writer, img1=img1, img2=img2, fused_img=fused_img, gt_img=gt_img,
                    gate_map=gate_map, # 🔥
                    epoch=epoch, batch_idx=batch_idx, mean=data_mean, std=data_std
                )
        
        # 计算平均
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1} 训练结果:")
        print(f"Total Loss: {train_metrics['total']:.4f}")
        print(f"Gate Mean (CLIP信任度): {train_metrics['gate_mean']:.4f}") # 越接近1越信赖CLIP，越接近0越信赖ResNet
        
        writer.add_scalar('Loss/Total_Train', train_metrics['total'], epoch)
        writer.add_scalar('Internal/Gate_Mean', train_metrics['gate_mean'], epoch) # 🔥
        
        # 保存图像
        if epoch % save_image_interval == 0:
            save_images_to_folder(
                img1=img1, img2=img2, fused_img=fused_img, gt_img=gt_img,
                focus_maps=focus_maps, gate_map=gate_map, # 🔥
                epoch=epoch, batch_idx=batch_idx, mean=data_mean, std=data_std,
                save_dir=image_save_dir
            )
        
        # 保存模型
        if train_metrics['total'] < best_train_loss:
            best_train_loss = train_metrics['total']
            torch.save(model.state_dict(), os.path.join(P['save_dir'], 'best_model.pth'))
            print(f"✅ 保存最佳模型")
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(P['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict()}, checkpoint_path)
    
    writer.close()
    print("训练完成！")


if __name__ == '__main__':
    P = {
        'batch_size': 8,
        'num_epochs': 80,
        'lr': 5e-4,
        'device': device,
        'use_amp': True,
        'detect_anomaly': False,
        'save_dir': './checkpoints_mfiwh_gated',
        'data_dir': './data/MFI-WHU',
        'num_workers': 2,
        'block_size': 32,
        'overlap': 4,
        'clip_patch_size': 32,
        'clip_stride': 16,
        # 'clip_weight': 0.3,  <-- 🔥 这一行必须删除或注释掉
        'train_backbone': True,
        'grad_clip': 5.0,
        'loss_weights': {
            'ssim': 2.0, 'l2': 1.0, 'fro': 0.1, 'focus_acc': 1.5, 'smooth': 0.1
        }
    }
    
    train_integrated_model(P)