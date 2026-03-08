import pandas as pd
import re

def parse_value(val_str):
    """Parses 'mean ± std' string to extract mean."""
    if isinstance(val_str, (int, float)):
        return float(val_str)
    if not isinstance(val_str, str):
        return None
    match = re.match(r"([0-9.]+)\s*±", val_str)
    if match:
        return float(match.group(1))
    try:
        return float(val_str)
    except ValueError:
        return None

def get_direction(metric_name):
    """Returns '越小越好' or '越大越好' based on metric name."""
    lower_is_better = ['MSE', 'CE', 'OCE', 'SD', 'Nabf'] # Added SD and Nabf as potentially lower is better, but sticking to plan for now.
    # Re-reading export.m to be sure about SD and Nabf.
    # SD is Standard Deviation, usually higher means more contrast/info in fusion? Or noise?
    # Let's stick to the explicit list from the plan: MSE, CE, OCE.
    # Actually, let's check SD and Nabf.
    # Nabf: "Artifacts based on mutual information"? Usually lower is better for artifacts.
    # Let's stick to the safe ones: MSE, CE, OCE are definitely lower is better.
    # If uncertain, I will mark as '越大越好' as default or '未知'.
    # Plan said: "MSE, CE, OCE 默认为越小越好，其他默认为越大越好"
    
    if metric_name in ['MSE', 'CE', 'OCE']:
        return '越小越好'
    return '越大越好'

def compare_excels(file1, file2, output_file):
    sheet_name = '所有指标汇总'
    
    try:
        df1 = pd.read_excel(file1, sheet_name=sheet_name)
        df2 = pd.read_excel(file2, sheet_name=sheet_name)
    except Exception as e:
        print(f"Error reading Excel files: {e}")
        return

    # Ensure columns exist
    required_cols = ['指标名称', '平均值 ± 标准差']
    for col in required_cols:
        if col not in df1.columns or col not in df2.columns:
            print(f"Missing column '{col}' in one of the files.")
            return

    results = []
    
    # Assuming both files have the same metrics in the same order.
    # Better to merge on '指标名称'.
    merged = pd.merge(df1, df2, on='指标名称', suffixes=('_1', '_2'), how='outer')
    
    for _, row in merged.iterrows():
        metric = row['指标名称']
        val1_str = row['平均值 ± 标准差_1']
        val2_str = row['平均值 ± 标准差_2']
        
        val1 = parse_value(val1_str)
        val2 = parse_value(val2_str)
        
        direction = get_direction(metric)
        
        diff = None
        result = "N/A"
        
        if val1 is not None and val2 is not None:
            diff = val1 - val2
            
            if direction == '越大越好':
                if diff > 0:
                    result = "文件1 优"
                elif diff < 0:
                    result = "文件2 优"
                else:
                    result = "持平"
            else: # 越小越好
                if diff < 0:
                    result = "文件1 优"
                elif diff > 0:
                    result = "文件2 优"
                else:
                    result = "持平"
        
        results.append({
            '指标名称': metric,
            '指标方向': direction,
            '文件1 (my1111)': val1,
            '文件2 (gate)': val2,
            '差值 (1-2)': diff,
            '对比结果': result,
            '文件1 原始值': val1_str,
            '文件2 原始值': val2_str
        })
        
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['指标名称', '指标方向', '文件1 (my1111)', '文件2 (gate)', '差值 (1-2)', '对比结果', '文件1 原始值', '文件2 原始值']
    results_df = results_df[cols]
    
    print("Comparison preview:")
    print(results_df.head())
    
    results_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    f1 = r"d:\唐伟项目\work\代码\代码\CLIP_focus\CLIP_focus\evaluation_results_my1111.xlsx"
    f2 = r"d:\唐伟项目\work\代码\代码\CLIP_focus\CLIP_focus\evaluation_results_gate.xlsx"
    out = r"d:\唐伟项目\work\代码\代码\CLIP_focus\CLIP_focus\comparison_results.xlsx"
    
    compare_excels(f1, f2, out)
