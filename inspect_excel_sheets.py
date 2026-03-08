import pandas as pd

file1 = r"d:\唐伟项目\work\代码\代码\CLIP_focus\CLIP_focus\evaluation_results_my1111.xlsx"
file2 = r"d:\唐伟项目\work\代码\代码\CLIP_focus\CLIP_focus\evaluation_results_gate.xlsx"

xl1 = pd.ExcelFile(file1)
xl2 = pd.ExcelFile(file2)

print("File 1 Sheets:", xl1.sheet_names)
print("File 2 Sheets:", xl2.sheet_names)

# Try reading the first sheet again without headers if it's empty
df1 = pd.read_excel(file1, sheet_name=xl1.sheet_names[0], header=None)
print("File 1 Head (header=None):\n", df1.head())

df2 = pd.read_excel(file2, sheet_name=xl2.sheet_names[0], header=None)
print("File 2 Head (header=None):\n", df2.head())
