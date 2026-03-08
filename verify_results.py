import pandas as pd

file = r"d:\唐伟项目\work\代码\代码\CLIP_focus\CLIP_focus\comparison_results.xlsx"

try:
    df = pd.read_excel(file)
    print("Output File Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("First 5 rows:\n", df.head())
    
    # Check if '指标方向' is correct for MSE
    mse_row = df[df['指标名称'] == 'MSE']
    if not mse_row.empty:
        print("\nMSE Row:\n", mse_row)
    
except Exception as e:
    print(f"Error reading output file: {e}")
