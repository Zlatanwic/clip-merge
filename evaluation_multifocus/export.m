% export_results_with_labels.m
% 导出带标识的评估结果到Excel（包含所有指标）

clear;
clc;

fprintf('正在加载评估结果...\n');
load Q_ave.mat;
load Q_std.mat;
load Q.mat;

% 指标名称对照表（根据evaluateSingle.m中的顺序，共38个指标）
metric_names = {
    'Q_MI',           % 1: 归一化互信息 (Normalized Mutual Information)
    'Q_TE',           % 2: Tsallis熵 (Tsallis Entropy)
    'Q_NCIE',         % 3: Wang-NCIE
    'Q_G',            % 4: Xydeas边缘 (Xydeas Edge)
    'Q_M',            % 5: PWW
    'Q_SF',           % 6: 空间频率 (Spatial Frequency)
    'Q_P',            % 7: 相位一致性 (Phase Congruency)
    'Q_s',            % 8: Piella结构相似性 (Piella Structural)
    'Q_C',            % 9: Cvejic
    'Q_Y',            % 10: Yang
    'Q_CV',           % 11: Chen-Varshney
    'Q_CB',           % 12: Chen-Blum
    'EN',             % 13: 熵 (Entropy)
    'LMI',            % 14: 局部互信息 (Local Mutual Information)
    'FMI',            % 15: 融合互信息 (Fusion Mutual Information)
    'Q_W',            % 16: Piella Q_W
    'Q_E',            % 17: Piella Q_E
    'VIFF',           % 18: 视觉信息保真度 (Visual Information Fidelity)
    'VIF_img1',       % 19: VIF for img1 (已禁用)
    'VIF_img2',       % 20: VIF for img2 (已禁用)
    'SCD',            % 21: 空间相关系数 (Spatial Correlation Deviation)
    'SD',             % 22: 标准差 (Standard Deviation)
    'AG',             % 23: 平均梯度 (Average Gradient)
    'CC',             % 24: 相关系数 (Correlation Coefficient)
    'PSNR',           % 25: 峰值信噪比 (Peak Signal-to-Noise Ratio)
    'MS-SSIM',        % 26: 多尺度结构相似性 (Multi-Scale SSIM)
    'VIF',            % 27: 视觉信息保真度 (Visual Information Fidelity)
    'MS_SSIM',        % 28: 多曝光MS-SSIM (Multi-Exposure MS-SSIM)
    'EI',             % 29: 边缘强度 (Edge Intensity)
    'OCE',            % 30: 总体交叉熵 (Overall Cross Entropy) (已禁用)
    'FMI_w',          % 31: FMI小波 (FMI Wavelet) (已禁用)
    'FMI_dct',        % 32: FMI DCT (已禁用)
    'Nabf',           % 33: Nabf
    'DF',             % 34: 清晰度 (Definition)
    'MI2',            % 35: MI2
    'CE',             % 36: 交叉熵 (Cross Entropy)
    'NMI',            % 37: 归一化互信息 (Normalized Mutual Information)
    'MSE'             % 38: 均方误差 (Mean Squared Error)
};

% 提取数据
results_ave = squeeze(Q_ave);  % 平均值
results_std = squeeze(Q_std);    % 标准差
results_detail = squeeze(Q);    % 详细结果 (40张图像)

num_metrics = length(results_ave);
num_images = size(results_detail, 1);

fprintf('检测到 %d 个指标，%d 张图像\n', num_metrics, num_images);

% ========== 工作表1: 平均值和标准差汇总（所有指标）==========
fprintf('正在导出平均值和标准差（所有 %d 个指标）...\n', num_metrics);

% 创建表格数据
summary_data = cell(num_metrics + 1, 4);
summary_data{1, 1} = '指标编号';
summary_data{1, 2} = '指标名称';
summary_data{1, 3} = '平均值 (Mean)';
summary_data{1, 4} = '标准差 (Std)';

for i = 1:num_metrics
    summary_data{i+1, 1} = i;
    if i <= length(metric_names)
        summary_data{i+1, 2} = metric_names{i};
    else
        summary_data{i+1, 2} = sprintf('Metric_%d', i);
    end
    
    if ~isnan(results_ave(i))
        summary_data{i+1, 3} = results_ave(i);
        summary_data{i+1, 4} = results_std(i);
    else
        summary_data{i+1, 3} = 'N/A';
        summary_data{i+1, 4} = 'N/A';
    end
end

% ========== 工作表2: 每张图像的详细结果（所有指标）==========
fprintf('正在导出详细结果（%d张图像 × %d个指标）...\n', num_images, num_metrics);

% 创建详细数据表格
detail_data = cell(num_images + 1, num_metrics + 1);
detail_data{1, 1} = '图像编号';

% 第一行：所有指标名称
for j = 1:num_metrics
    if j <= length(metric_names)
        detail_data{1, j+1} = metric_names{j};
    else
        detail_data{1, j+1} = sprintf('Metric_%d', j);
    end
end

% 填充数据
for i = 1:num_images
    detail_data{i+1, 1} = i;
    for j = 1:num_metrics
        if ~isnan(results_detail(i, j))
            detail_data{i+1, j+1} = results_detail(i, j);
        else
            detail_data{i+1, j+1} = 'N/A';
        end
    end
end

% ========== 工作表3: 所有指标汇总（与工作表1相同，但格式不同）==========
fprintf('正在导出所有指标汇总...\n');

% 创建汇总表格（转置格式，便于查看）
all_metrics_data = cell(num_metrics + 1, 3);
all_metrics_data{1, 1} = '指标编号';
all_metrics_data{1, 2} = '指标名称';
all_metrics_data{1, 3} = '平均值 ± 标准差';

for i = 1:num_metrics
    all_metrics_data{i+1, 1} = i;
    if i <= length(metric_names)
        all_metrics_data{i+1, 2} = metric_names{i};
    else
        all_metrics_data{i+1, 2} = sprintf('Metric_%d', i);
    end
    
    if ~isnan(results_ave(i))
        all_metrics_data{i+1, 3} = sprintf('%.6f ± %.6f', results_ave(i), results_std(i));
    else
        all_metrics_data{i+1, 3} = 'N/A';
    end
end

% ========== 导出到Excel ==========
excel_filename = 'evaluation_results_labeled.xlsx';
fprintf('正在导出到Excel文件: %s\n', excel_filename);

try
    % 删除已存在的文件（如果存在）
    if exist(excel_filename, 'file')
        delete(excel_filename);
    end
    
    % 写入工作表1: 平均值和标准差（所有指标）
    xlswrite(excel_filename, summary_data, '平均值和标准差');
    
    % 写入工作表2: 详细结果（所有指标）
    xlswrite(excel_filename, detail_data, '详细结果（40张图像）');
    
    % 写入工作表3: 所有指标汇总
    xlswrite(excel_filename, all_metrics_data, '所有指标汇总');
    
    fprintf('\n✅ 导出成功！\n');
    fprintf('文件位置: %s\n', fullfile(pwd, excel_filename));
    fprintf('\nExcel文件包含3个工作表：\n');
    fprintf('  1. 平均值和标准差 - 所有 %d 个指标的平均值和标准差\n', num_metrics);
    fprintf('  2. 详细结果（40张图像） - 每张图像的每个指标值（%d列）\n', num_metrics);
    fprintf('  3. 所有指标汇总 - 所有 %d 个指标的平均值±标准差\n', num_metrics);
    
catch ME
    fprintf('❌ 导出失败: %s\n', ME.message);
    fprintf('尝试使用writetable函数...\n');
    
    % 使用writetable替代（MATLAB R2013b+）
    try
        summary_table = cell2table(summary_data(2:end, :), 'VariableNames', summary_data(1, :));
        writetable(summary_table, excel_filename, 'Sheet', '平均值和标准差');
        
        detail_table = cell2table(detail_data(2:end, :), 'VariableNames', detail_data(1, :));
        writetable(detail_table, excel_filename, 'Sheet', '详细结果（40张图像）');
        
        all_metrics_table = cell2table(all_metrics_data(2:end, :), 'VariableNames', all_metrics_data(1, :));
        writetable(all_metrics_table, excel_filename, 'Sheet', '所有指标汇总');
        
        fprintf('✅ 使用writetable导出成功！\n');
    catch ME2
        fprintf('❌ writetable也失败: %s\n', ME2.message);
    end
end

% ========== 在命令窗口显示所有结果 ==========
fprintf('\n========================================\n');
fprintf('    所有评估指标（平均值 ± 标准差）\n');
fprintf('========================================\n\n');

valid_count = 0;
for i = 1:num_metrics
    if ~isnan(results_ave(i))
        valid_count = valid_count + 1;
        metric_name = metric_names{min(i, length(metric_names))};
        fprintf('  [%2d] %-15s: %12.6f ± %12.6f\n', i, metric_name, ...
                results_ave(i), results_std(i));
    end
end

fprintf('\n有效指标数量: %d / %d\n', valid_count, num_metrics);
fprintf('结果已保存到: %s\n', excel_filename);