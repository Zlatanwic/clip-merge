import pandas as pd

file1 = r"d:\唐伟项目\work\代码\代码\CLIP_focus\CLIP_focus\evaluation_results_my1111.xlsx"
file2 = r"d:\唐伟项目\work\代码\代码\CLIP_focus\CLIP_focus\evaluation_results_gate.xlsx"

df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

print("File 1 Shape:", df1.shape)
print("File 1 Columns:", df1.columns.tolist())
print("File 1 Head:\n", df1.head())

print("-" * 20)

print("File 2 Shape:", df2.shape)
print("File 2 Columns:", df2.columns.tolist())
print("File 2 Head:\n", df2.head())
