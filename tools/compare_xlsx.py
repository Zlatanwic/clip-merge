#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较两个xlsx表格中的38个指标
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_xlsx_data(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """加载xlsx文件数据"""
    try:
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            # 尝试读取第一个工作表
            df = pd.read_excel(file_path, sheet_name=0)
        return df
    except Exception as e:
        print(f"❌ 读取文件 {file_path} 失败: {e}")
        raise


def find_metrics_columns(df: pd.DataFrame) -> List[str]:
    """查找指标列（排除索引列和图像名称列）"""
    # 通常第一列是图像名称或索引，其余是指标
    cols = df.columns.tolist()
    # 排除常见的非指标列
    non_metric_cols = ['图像名称', '图像', 'Image', 'Index', 'index', '序号', '编号']
    metric_cols = [col for col in cols if col not in non_metric_cols]
    return metric_cols


def compare_metrics(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    file1_name: str,
    file2_name: str,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    比较两个数据框中的指标
    
    Args:
        df1: 第一个数据框
        df2: 第二个数据框
        file1_name: 第一个文件名（用于显示）
        file2_name: 第二个文件名（用于显示）
        output_file: 输出文件路径（可选）
    
    Returns:
        比较结果数据框
    """
    # 获取指标列
    metrics1 = find_metrics_columns(df1)
    metrics2 = find_metrics_columns(df2)
    
    # 找到共同的指标
    common_metrics = sorted(set(metrics1) & set(metrics2))
    
    print(f"\n📊 数据概览:")
    print(f"  文件1 ({file1_name}): {len(df1)} 行, {len(metrics1)} 个指标")
    print(f"  文件2 ({file2_name}): {len(df2)} 行, {len(metrics2)} 个指标")
    print(f"  共同指标: {len(common_metrics)} 个")
    
    if len(common_metrics) == 0:
        print("❌ 未找到共同的指标列！")
        return pd.DataFrame()
    
    # 如果用户指定了38个指标，检查是否匹配
    if len(common_metrics) != 38:
        print(f"⚠️  警告: 找到 {len(common_metrics)} 个共同指标，不是38个")
        print(f"   文件1独有指标: {set(metrics1) - set(metrics2)}")
        print(f"   文件2独有指标: {set(metrics2) - set(metrics1)}")
    
    # 准备比较结果
    comparison_results = []
    
    for metric in common_metrics:
        # 获取两个文件中的该指标数据
        values1 = pd.to_numeric(df1[metric], errors='coerce').dropna()
        values2 = pd.to_numeric(df2[metric], errors='coerce').dropna()
        
        if len(values1) == 0 or len(values2) == 0:
            continue
        
        # 计算统计量
        mean1 = values1.mean()
        mean2 = values2.mean()
        std1 = values1.std()
        std2 = values2.std()
        min1 = values1.min()
        min2 = values2.min()
        max1 = values1.max()
        max2 = values2.max()
        
        # 计算差异
        mean_diff = mean2 - mean1
        mean_diff_pct = (mean_diff / mean1 * 100) if mean1 != 0 else 0
        
        # 判断哪个更好（通常指标越大越好，但有些指标越小越好，这里假设越大越好）
        # 可以根据实际需求调整
        better = "文件2" if mean2 > mean1 else "文件1"
        
        comparison_results.append({
            '指标名称': metric,
            f'{file1_name}_平均值': mean1,
            f'{file2_name}_平均值': mean2,
            '平均值差异': mean_diff,
            '平均值差异(%)': mean_diff_pct,
            f'{file1_name}_标准差': std1,
            f'{file2_name}_标准差': std2,
            f'{file1_name}_最小值': min1,
            f'{file2_name}_最小值': min2,
            f'{file1_name}_最大值': max1,
            f'{file2_name}_最大值': max2,
            '更好': better
        })
    
    result_df = pd.DataFrame(comparison_results)
    
    # 按平均值差异排序
    result_df = result_df.sort_values('平均值差异', ascending=False)
    
    # 打印摘要
    print(f"\n📈 比较结果摘要:")
    print(f"  文件2优于文件1的指标数: {len(result_df[result_df['更好'] == '文件2'])}")
    print(f"  文件1优于文件2的指标数: {len(result_df[result_df['更好'] == '文件1'])}")
    print(f"\n  平均差异最大的5个指标（文件2 - 文件1）:")
    top5 = result_df.head(5)
    for _, row in top5.iterrows():
        print(f"    {row['指标名称']}: {row['平均值差异']:.6f} ({row['平均值差异(%)']:.2f}%)")
    
    print(f"\n  平均差异最小的5个指标（文件2 - 文件1）:")
    bottom5 = result_df.tail(5)
    for _, row in bottom5.iterrows():
        print(f"    {row['指标名称']}: {row['平均值差异']:.6f} ({row['平均值差异(%)']:.2f}%)")
    
    # 保存结果
    if output_file:
        result_df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ 比较结果已保存到: {output_file}")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="比较两个xlsx表格中的指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python compare_xlsx.py file1.xlsx file2.xlsx
  python compare_xlsx.py file1.xlsx file2.xlsx --sheet "平均值和标准差"
  python compare_xlsx.py file1.xlsx file2.xlsx --output comparison_result.xlsx
        """
    )
    parser.add_argument(
        "file1",
        type=str,
        help="第一个xlsx文件路径"
    )
    parser.add_argument(
        "file2",
        type=str,
        help="第二个xlsx文件路径"
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="要读取的工作表名称（默认读取第一个工作表）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出比较结果的xlsx文件路径（默认: comparison_result.xlsx）"
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    file1_path = Path(args.file1)
    file2_path = Path(args.file2)
    
    if not file1_path.exists():
        print(f"❌ 文件不存在: {file1_path}")
        return
    
    if not file2_path.exists():
        print(f"❌ 文件不存在: {file2_path}")
        return
    
    # 加载数据
    print(f"📂 正在加载文件...")
    df1 = load_xlsx_data(str(file1_path), args.sheet)
    df2 = load_xlsx_data(str(file2_path), args.sheet)
    
    # 确定输出文件
    if args.output:
        output_file = args.output
    else:
        output_file = "comparison_result.xlsx"
    
    # 执行比较
    result_df = compare_metrics(
        df1, df2,
        file1_path.stem,
        file2_path.stem,
        output_file
    )
    
    # 显示完整结果
    print(f"\n📋 完整比较结果:")
    print(result_df.to_string(index=False))
    
    print(f"\n✅ 完成！")


if __name__ == "__main__":
    main()
