"""
Download Stanford Cars dataset from Hugging Face and convert to required format
根据split json来组织图像，确保图像文件名和标签与split json一致
"""
import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import json
from tqdm import tqdm
import scipy.io as sio
import numpy as np


# 输出目录
output_dir = Path("/home/wangzh/code-sapce/NLPrompt/data/stanford_cars_hf")
output_dir.mkdir(parents=True, exist_ok=True)

# 读取split json
split_json_path = output_dir / "split_zhou_StanfordCars.json"
print(f"Loading split json from {split_json_path}...")
with open(split_json_path, 'r') as f:
    split_data = json.load(f)

print(f"Split data - Train: {len(split_data['train'])}, Val: {len(split_data['val'])}, Test: {len(split_data['test'])}")

# 加载HF数据集
print("Loading Stanford Cars dataset from Hugging Face...")
ds = load_dataset("tanganke/stanford_cars")

print(f"HF dataset - Train: {len(ds['train'])}, Test: {len(ds['test'])}")

# 创建目录
cars_train_dir = output_dir / "cars_train"
cars_test_dir = output_dir / "cars_test"
devkit_dir = output_dir / "devkit"

cars_train_dir.mkdir(exist_ok=True)
cars_test_dir.mkdir(exist_ok=True)
devkit_dir.mkdir(exist_ok=True)

# 获取所有类别名称
all_classnames = set()
for item in split_data['train'] + split_data['val'] + split_data['test']:
    all_classnames.add(item[2])
classnames = sorted(list(all_classnames))
print(f"Total classes: {len(classnames)}")

# 创建classname到索引的映射
classname_to_idx = {name: idx for idx, name in enumerate(classnames)}

# ===== 处理训练集 =====
print("\n=== Processing Training Set ===")
# 创建从split json文件名到HF数据集索引的映射
# split json中的文件名格式是 "cars_train/05266.jpg"，提取数字部分
train_file_to_hf_idx = {}
for hf_idx in range(len(ds['train'])):
    train_file_to_hf_idx[hf_idx] = hf_idx

# 创建用于.mat文件的注释数组
train_annotations = []

# 遍历split json中的训练集和验证集条目
for item in tqdm(split_data['train'] + split_data['val'], desc="Processing train+val"):
    img_path, label, classname = item
    # 从路径中提取文件名，例如 "cars_train/05266.jpg" -> "05266.jpg"
    filename = os.path.basename(img_path)
    # 提取索引号，例如 "05266.jpg" -> 5266
    img_idx = int(filename.split('.')[0]) - 1  # 转换为0-based索引
    
    # 从HF数据集获取对应的图像
    if img_idx < len(ds['train']):
        hf_item = ds['train'][img_idx]
        img = hf_item['image']
        
        # 保存图像
        save_path = cars_train_dir / filename
        img.save(save_path)
        
        # 添加注释信息（matlab使用1-based索引）
        train_annotations.append({
            'fname': filename,
            'class': label + 1,  # matlab使用1-based索引
            'classname': classname
        })

# ===== 处理测试集 =====
print("\n=== Processing Test Set ===")
test_annotations = []

for item in tqdm(split_data['test'], desc="Processing test"):
    img_path, label, classname = item
    # 从路径中提取文件名，例如 "cars_test/00001.jpg" -> "00001.jpg"
    filename = os.path.basename(img_path)
    # 提取索引号，例如 "00001.jpg" -> 1
    img_idx = int(filename.split('.')[0]) - 1  # 转换为0-based索引
    
    # 从HF数据集获取对应的图像
    if img_idx < len(ds['test']):
        hf_item = ds['test'][img_idx]
        img = hf_item['image']
        
        # 保存图像
        save_path = cars_test_dir / filename
        img.save(save_path)
        
        # 添加注释信息（matlab使用1-based索引）
        test_annotations.append({
            'fname': filename,
            'class': label + 1,  # matlab使用1-based索引
            'classname': classname
        })

# ===== 创建.mat文件 =====
print("\n=== Creating .mat files ===")

# 创建cars_meta.mat
meta_data = {
    'class_names': np.array([[name] for name in classnames], dtype=object)
}
sio.savemat(devkit_dir / 'cars_meta.mat', meta_data)
print(f"Saved {devkit_dir / 'cars_meta.mat'}")

# 创建cars_train_annos.mat
train_annos_structured = np.zeros((len(train_annotations),), dtype=[
    ('fname', 'O'),
    ('class', 'O')
])
for i, anno in enumerate(train_annotations):
    train_annos_structured[i] = (anno['fname'], np.array([[anno['class']]], dtype=np.uint8))

train_mat_data = {'annotations': train_annos_structured}
sio.savemat(devkit_dir / 'cars_train_annos.mat', train_mat_data)
print(f"Saved {devkit_dir / 'cars_train_annos.mat'} with {len(train_annotations)} annotations")

# 创建cars_test_annos_withlabels.mat
test_annos_structured = np.zeros((len(test_annotations),), dtype=[
    ('fname', 'O'),
    ('class', 'O')
])
for i, anno in enumerate(test_annotations):
    test_annos_structured[i] = (anno['fname'], np.array([[anno['class']]], dtype=np.uint8))

test_mat_data = {'annotations': test_annos_structured}
sio.savemat(output_dir / 'cars_test_annos_withlabels.mat', test_mat_data)
print(f"Saved {output_dir / 'cars_test_annos_withlabels.mat'} with {len(test_annotations)} annotations")

# 也复制到devkit目录
sio.savemat(devkit_dir / 'cars_test_annos_withlabels.mat', test_mat_data)
print(f"Saved {devkit_dir / 'cars_test_annos_withlabels.mat'}")

print("\n=== Dataset conversion completed! ===")
print(f"Train images: {len(train_annotations)}")
print(f"Test images: {len(test_annotations)}")
print(f"Total classes: {len(classnames)}")
    
 