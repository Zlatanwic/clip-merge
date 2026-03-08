import os
import re
from pathlib import Path


def batch_rename_files(directory, pattern=r'(\d+)\.jpg', dry_run=True):
    """
    批量重命名文件,将文件名中的数字部分补零为3位
    
    Args:
        directory: 目标目录路径
        pattern: 正则表达式模式,用于匹配文件名
        dry_run: 是否为预览模式(True=仅预览,False=实际重命名)
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"错误: 目录不存在 - {directory}")
        return
    
    # 获取所有匹配的文件
    files_to_rename = []
    for file in directory.iterdir():
        if file.is_file():
            match = re.match(pattern, file.name)
            if match:
                old_number = int(match.group(1))
                new_name = f"{old_number:03d}.jpg"
                files_to_rename.append((file, new_name, old_number))
    
    # 按原始数字排序
    files_to_rename.sort(key=lambda x: x[2])
    
    if not files_to_rename:
        print("没有找到匹配的文件")
        return
    
    print(f"找到 {len(files_to_rename)} 个文件需要重命名\n")
    
    if dry_run:
        print("=" * 60)
        print("预览模式 - 以下是将要执行的重命名操作:")
        print("=" * 60)
        for old_file, new_name, _ in files_to_rename:
            print(f"{old_file.name:20s} -> {new_name}")
        print("=" * 60)
        print(f"\n总计: {len(files_to_rename)} 个文件")
        print("\n如果确认无误,请将 dry_run=False 来执行实际重命名")
    else:
        print("=" * 60)
        print("开始重命名...")
        print("=" * 60)
        
        success_count = 0
        error_count = 0
        
        for old_file, new_name, _ in files_to_rename:
            new_file = old_file.parent / new_name
            
            # 如果新文件名已存在且不是同一个文件,跳过
            if new_file.exists() and new_file != old_file:
                print(f"跳过 {old_file.name} - 目标文件已存在: {new_name}")
                error_count += 1
                continue
            
            try:
                old_file.rename(new_file)
                print(f"✓ {old_file.name:20s} -> {new_name}")
                success_count += 1
            except Exception as e:
                print(f"✗ 重命名失败 {old_file.name}: {e}")
                error_count += 1
        
        print("=" * 60)
        print(f"重命名完成!")
        print(f"成功: {success_count} 个文件")
        if error_count > 0:
            print(f"失败: {error_count} 个文件")
        print("=" * 60)


if __name__ == "__main__":
    # 目标目录
    target_dir = r"D:\唐伟项目\work\代码\代码\CLIP_focus\CLIP_focus\data\MFI-WHU\source_2"
    
    # 第一步: 预览模式,查看将要执行的重命名操作
    print("第一步: 预览重命名操作\n")
    batch_rename_files(target_dir, dry_run=True)
    
    # 第二步: 执行实际重命名
    print("\n\n第二步: 执行实际重命名\n")
    batch_rename_files(target_dir, dry_run=False)
