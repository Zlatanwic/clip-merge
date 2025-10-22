import os
import shutil
import random

def split_dataset(data_dir, train_ratio=0.8, seed=42):
    """
    Split MFI-WHU dataset into train and val
    """
    random.seed(seed)
    
    # Create train and val directories
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    for split_dir in [train_dir, val_dir]:
        for subdir in ['source_1', 'source_2', 'full_clear']:
            os.makedirs(os.path.join(split_dir, subdir), exist_ok=True)
    
    # Get all image IDs
    source1_dir = os.path.join(data_dir, 'source_1')
    source2_dir = os.path.join(data_dir, 'source_2')
    full_clear_dir = os.path.join(data_dir, 'full_clear')
    
    source1_files = set([f.split('.')[0] for f in os.listdir(source1_dir) if f.endswith('.jpg')])
    source2_files = set([f.split('.')[0] for f in os.listdir(source2_dir) if f.endswith('.jpg')])
    full_clear_files = set([f.split('.')[0] for f in os.listdir(full_clear_dir) if f.endswith('.jpg')])
    
    common_ids = source1_files & source2_files & full_clear_files
    image_ids = sorted(list(common_ids))
    
    # Shuffle and split
    random.shuffle(image_ids)
    num_train = int(len(image_ids) * train_ratio)
    train_ids = image_ids[:num_train]
    val_ids = image_ids[num_train:]
    
    print(f"Total images: {len(image_ids)}")
    print(f"Train images: {len(train_ids)}")
    print(f"Val images: {len(val_ids)}")
    
    # Copy files
    for img_id in train_ids:
        for subdir in ['source_1', 'source_2', 'full_clear']:
            src = os.path.join(data_dir, subdir, f'{img_id}.jpg')
            dst = os.path.join(train_dir, subdir, f'{img_id}.jpg')
            shutil.copy2(src, dst)
    
    for img_id in val_ids:
        for subdir in ['source_1', 'source_2', 'full_clear']:
            src = os.path.join(data_dir, subdir, f'{img_id}.jpg')
            dst = os.path.join(val_dir, subdir, f'{img_id}.jpg')
            shutil.copy2(src, dst)
    
    print("Dataset split completed!")

if __name__ == '__main__':
    data_dir = 'data/MFI-WHU'
    split_dataset(data_dir, train_ratio=0.8)
