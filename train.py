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

# 导入数据集（使用修改后的仅训练集加载器）和模型
from datasets import get_mfiwh_train_loader  # 仅导入训练集加载器
from modelv2 import MultiFocusFusionModel

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")


# -------------------------- 损失函数（保持原有） --------------------------
def _torch_fspecial_gauss(size, sigma, device):
    """生成高斯核，模拟MATLAB的fspecial函数"""
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    x = torch.tensor(x_data, dtype=torch.float32, device=device)
    y = torch.tensor(y_data, dtype=torch.float32, device=device)
    g = torch.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / torch.sum(g)

def SSIM_LOSS(img1, img2, size=11, sigma=1.5):
    """结构相似结构相似性损失函数：1 - SSIM"""
    device = img1.device
    window = _torch_fspecial_gauss(size, sigma, device)
    window = window.view(1, 1, size, size).repeat(img1.size(1), 1, 1, 1)
    
    K1 = 0.01
    K2 = 0.03
    L = 1  # 图像已归一化到[0,1]
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    # 计算局部均值
    mu1 = F.conv2d(img1, window, stride=1, padding=size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, stride=1, padding=size//2, groups=img2.size(1))
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1 * img1, window, stride=1, padding=size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, stride=1, padding=size//2, groups=img1.size(1)) - mu1_mu2
    
    # 计算SSIM并返回损失
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1 - torch.mean(ssim_map)

def L2_LOSS(batchimg):
    """L2损失（平方损失）"""
    l2_norm = torch.norm(batchimg, p=2, dim=[1, 2]) / (batchimg.shape[1] * batchimg.shape[2])
    return torch.mean(l2_norm)

def Fro_LOSS(batchimg):
    """Frobenius损失（矩阵L2范数平方）"""
    fro_norm = torch.square(torch.norm(batchimg, p='fro', dim=[1, 2])) 
    fro_norm = fro_norm / (batchimg.shape[1] * batchimg.shape[2])
    return torch.mean(fro_norm)


# -------------------------- 辅助损失函数（保持不变） --------------------------
def create_smoothness_loss(focus_maps):
    """平滑性损失：约束聚焦图空间连续性"""
    grad_x = torch.abs(focus_maps[:, :, :, 1:] - focus_maps[:, :, :, :-1])
    grad_y = torch.abs(focus_maps[:, :, 1:, :] - focus_maps[:, :, :-1, :])
    return torch.mean(grad_x) + torch.mean(grad_y)

def create_focus_accuracy_loss(focus_maps, source1, source2, gt):
    """聚焦准确性损失：匹配理想聚焦图"""
    with autocast(enabled=False):
        focus_maps = focus_maps.to(torch.float32)
        source1 = source1.to(torch.float32)
        source2 = source2.to(torch.float32)

        def calculate_sharpness(img):
            """计算图像清晰度（拉普拉斯算子）"""
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


# -------------------------- 辅助函数（图像日志、保存、归一化检查） --------------------------
def check_normalization(tensor, name, step, writer):
    """检查张量归一化情况并记录到TensorBoard"""
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

def save_images_to_folder(img1, img2, fused_img, gt_img, epoch, batch_idx, mean, std, save_dir):
    """保存训练过程中的图像（含反归一化）"""
    epoch_dir = os.path.join(save_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    def denormalize(tensor):
        """反归一化：恢复到[0,1]范围"""
        tensor = tensor * std.view(3, 1, 1) + mean.view(3, 1, 1)
        return torch.clamp(tensor, 0.0, 1.0)
    
    for i in range(img1.size(0)):
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

def log_images_to_tensorboard(writer, img1, img2, fused_img, gt_img, epoch, batch_idx, mean, std):
    """将图像记录到TensorBoard（实时查看训练效果）"""
    sample_idx = 0  # 取批次中第一个样本展示
    img1_sample = img1[sample_idx]
    img2_sample = img2[sample_idx]
    fused_sample = fused_img[sample_idx]
    gt_sample = gt_img[sample_idx]
    
    def denormalize(tensor):
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


# -------------------------- 训练主函数（移除所有验证逻辑） --------------------------
def train_integrated_model(P):
    detect_anomaly = P.get('detect_anomaly', False)
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    os.makedirs(P['save_dir'], exist_ok=True)
    
    # 创建融合图像保存目录
    image_save_dir = os.path.join(P['save_dir'], 'fusion_results')
    os.makedirs(image_save_dir, exist_ok=True)
    print(f"融合图像将保存至: {image_save_dir}")
    
    # 1. 加载MFI-WHU全量数据（无验证集，全部用于训练）
    print("\n[1/5] 加载MFI-WHU全量训练集...")
    train_loader = get_mfiwh_train_loader(
        root_dir=P['data_dir'],
        batch_size=P['batch_size'],
        num_workers=P['num_workers']
    )
    print(f"✅ 全量训练集加载完成：共{len(train_loader.dataset)}张样本，{len(train_loader)}个批次")
    
    # 获取归一化参数（优先用数据集自带，无则用ImageNet默认）
    try:
        data_mean = torch.tensor(train_loader.dataset.mean, device=device)
        data_std = torch.tensor(train_loader.dataset.std, device=device)
        print(f"使用数据集自带归一化参数: mean={data_mean.tolist()}, std={data_std.tolist()}")
    except:
        data_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        data_std = torch.tensor([0.229, 0.224, 0.225], device=device)
        print(f"使用ImageNet默认归一化参数: mean={data_mean.tolist()}, std={data_std.tolist()}")
    
    # 2. 初始化多聚焦融合模型
    print("\n[2/5] 初始化多聚焦融合模型...")
    model = MultiFocusFusionModel(
        block_size=P['block_size'],
        overlap=P['overlap']
    ).to(P['device'])
    print(f"✅ 模型初始化完成，参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 3. 配置优化器和学习率调度器
    print("\n[3/5] 设置优化器与学习率调度...")
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=P['lr'])
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)  # 每20轮学习率减半
    print(f"✅ 优化器: Adam (lr={P['lr']}), 调度器: StepLR (step_size=20, gamma=0.5)")
    amp_device_available = torch.cuda.is_available() and 'cuda' in str(P['device']).lower()
    requested_amp = P.get('use_amp', amp_device_available)
    use_amp = requested_amp and amp_device_available
    if requested_amp and not amp_device_available:
        print("⚠️ 检测到非 CUDA 设备，自动混合精度已自动关闭")
    scaler = GradScaler(enabled=use_amp)
    print(f"⚙️ 自动混合精度: {'开启' if use_amp else '关闭'}")
    
    # 4. 初始化TensorBoard日志
    log_dir = os.path.join(P['save_dir'], 'tensorboard_logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"✅ TensorBoard日志目录: {log_dir} (运行: tensorboard --logdir {log_dir})")
    
    
    # 5. 训练循环（仅训练，无验证）
    print("\n[4/5] 开始全量数据训练...")
    best_train_loss = float('inf')  # 跟踪训练集最佳损失（无验证损失）
    log_interval = 5  # 每5个批次记录一次TensorBoard图像
    save_image_interval = 1  # 每1个epoch保存一次图像
    
    for epoch in range(P['num_epochs']):
        model.train()  # 确保模型处于训练模式
        train_metrics = {'total': 0, 'ssim': 0, 'l2': 0, 'fro': 0, 'focus_acc': 0, 'smooth': 0}
        
        # 遍历训练集批次
        for batch_idx, (img1, img2, gt_img) in enumerate(tqdm(
            train_loader, desc=f'Epoch {epoch+1}/{P["num_epochs"]} (Train)'
        )):
            img1, img2, gt_img = img1.to(P['device']), img2.to(P['device']), gt_img.to(P['device'])
            
            # 定期检查输入/输出归一化情况（避免数值异常）
            if epoch % 10 == 0 and batch_idx == 0:
                check_normalization(img1, 'img1', epoch, writer)
                check_normalization(img2, 'img2', epoch, writer)
                check_normalization(gt_img, 'gt_img', epoch, writer)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=use_amp):
                # 前向传播：生成融合图像和聚焦图
                fused_img, focus_maps = model(img1, img2)
                
                # 检查融合图像归一化情况
                if epoch % 10 == 0 and batch_idx == 0:
                    check_normalization(fused_img, 'fused_img', epoch, writer)
                
                # 计算各损失分量
                loss_ssim = SSIM_LOSS(fused_img, gt_img)
                loss_l2 = L2_LOSS(fused_img - gt_img)
                loss_fro = Fro_LOSS(fused_img - gt_img)
                loss_focus_acc = create_focus_accuracy_loss(focus_maps, img1, img2, gt_img)
                loss_smooth = create_smoothness_loss(focus_maps)
                
                # 总损失（权重与原逻辑保持一致）
                total_loss = (1.0 * loss_ssim +
                              0.5 * loss_l2 +
                              0.3 * loss_fro +
                              2.0 * loss_focus_acc +
                              0.2 * loss_smooth)
            
            # 反向传播与参数更新
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # 累计训练指标（用于计算epoch平均损失）
            train_metrics['total'] += total_loss.item()
            train_metrics['ssim'] += loss_ssim.item()
            train_metrics['l2'] += loss_l2.item()
            train_metrics['fro'] += loss_fro.item()
            train_metrics['focus_acc'] += loss_focus_acc.item()
            train_metrics['smooth'] += loss_smooth.item()
            
            # 定期记录图像到TensorBoard
            if batch_idx % log_interval == 0:
                log_images_to_tensorboard(
                    writer=writer,
                    img1=img1,
                    img2=img2,
                    fused_img=fused_img,
                    gt_img=gt_img,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    mean=data_mean,
                    std=data_std
                )
        
        # 计算epoch平均训练损失
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        
        # 学习率调度器更新
        scheduler.step()
        
        # 打印当前epoch训练结果
        print(f"\nEpoch {epoch+1} 训练结果:")
        print(f"训练总损失: {train_metrics['total']:.4f}")
        print(f"  - SSIM损失: {train_metrics['ssim']:.4f}")
        print(f"  - L2损失: {train_metrics['l2']:.4f}")
        print(f"  - Frobenius损失: {train_metrics['fro']:.4f}")
        print(f"  - 聚焦准确性损失: {train_metrics['focus_acc']:.4f}")
        print(f"  - 平滑性损失: {train_metrics['smooth']:.4f}")
        print(f"当前学习率: {scheduler.get_last_lr()[0]:.6f}")
        
        # 记录训练指标到TensorBoard
        writer.add_scalar('Loss/Total_Train', train_metrics['total'], epoch)
        writer.add_scalar('Loss/SSIM_Train', train_metrics['ssim'], epoch)
        writer.add_scalar('Loss/L2_Train', train_metrics['l2'], epoch)
        writer.add_scalar('Loss/Frobenius_Train', train_metrics['fro'], epoch)
        writer.add_scalar('Loss/Focus_Acc_Train', train_metrics['focus_acc'], epoch)
        writer.add_scalar('Loss/Smooth_Train', train_metrics['smooth'], epoch)
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
        
        # 保存当前epoch的融合图像（便于可视化训练过程）
        if epoch % save_image_interval == 0:
            save_images_to_folder(
                img1=img1,  # 取最后一个批次的图像（也可改为固定批次）
                img2=img2,
                fused_img=fused_img,
                gt_img=gt_img,
                epoch=epoch,
                batch_idx=batch_idx,
                mean=data_mean,
                std=data_std,
                save_dir=image_save_dir
            )
        
        # 保存训练集损失最优的模型
        if train_metrics['total'] < best_train_loss:
            best_train_loss = train_metrics['total']
            torch.save(model.state_dict(), os.path.join(P['save_dir'], 'best_model.pth'))
            print(f"✅ 保存最佳模型（训练损失: {best_train_loss:.4f}）")
        
        # 定期保存训练检查点（每10个epoch）
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(P['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_train_loss': best_train_loss,
                'current_lr': scheduler.get_last_lr()[0]
            }, checkpoint_path)
            print(f"💾 保存第 {epoch+1} 轮检查点: {checkpoint_path}")
    
    # 训练结束：关闭日志写入器
    writer.close()
    print("\n" + "="*50)
    print("训练完成！")
    print(f"📁 最佳模型: {os.path.join(P['save_dir'], 'best_model.pth')}")
    print(f"📁 融合图像: {image_save_dir}")
    print(f"📁 TensorBoard日志: {log_dir}")
    print("="*50)


if __name__ == '__main__':
    # 训练参数配置（移除原验证相关参数）
    P = {
        'batch_size': 8,          # 批次大小（根据GPU显存调整）
        'num_epochs': 50,         # 训练轮数
        'lr': 5e-4,               # 初始学习率
        'device': device,         # 训练设备（cuda/cpu）
        'use_amp': True,          # 是否启用自动混合精度
        'detect_anomaly': False,  # 是否开启梯度异常检测
        'save_dir': './checkpoints_mfiwh_full',  # 模型/日志保存目录
        'data_dir': './data/MFI-WHU',            # MFI-WHU数据集根目录（含source_1/source_2/full_clear）
        'num_workers': 2,         # 数据加载线程数（建议不超过CPU核心数）
        'block_size': 32,         # 模型块大小（与modelv2.py保持一致）
        'overlap': 4              # 模型块重叠率（与modelv2.py保持一致）
    }
    
    # 确保保存目录存在
    os.makedirs(P['save_dir'], exist_ok=True)
    
    # 启动训练
    train_integrated_model(P)
    
