import pandas as pd

file1 = r"d:\唐伟项目\work\代码\代码\CLIP_focus\CLIP_focus\evaluation_results_my1111.xlsx"
sheet_name = '所有指标汇总'

df = pd.read_excel(file1, sheet_name=sheet_name)

print("Columns repr:", repr(df.columns.tolist()))

# Try accessing by index
print("Second column head:\n", df.iloc[:, 1].head())
print("Number of rows:", len(df))
