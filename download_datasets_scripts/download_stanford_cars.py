"""
Download Stanford Cars dataset from Hugging Face and convert to required format
"""
import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import json
from tqdm import tqdm
import scipy.io as sio


output_dir = Path("/home/wangzh/code-sapce/NLPrompt/data/stanford_cars")
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading Stanford Cars dataset from Hugging Face...")

ds = load_dataset("tanganke/stanford_cars")


cars_train_dir = output_dir / "cars_train"
cars_test_dir = output_dir / "cars_test"

cars_train_dir.mkdir(exist_ok=True)
cars_test_dir.mkdir(exist_ok=True)


for idx, item in enumerate(tqdm(ds['train'], desc="Train")):
    img = item['image']
    label = item['label']
    
    img_name = f"{idx+1:05d}.jpg"
    img_path = cars_train_dir / img_name
    img.save(img_path)


for idx, item in enumerate(tqdm(ds['test'], desc="Test")):
    img = item['image']
    label = item['label']
    
    img_name = f"{idx+1:05d}.jpg"
    img_path = cars_test_dir / img_name
    img.save(img_path)
    
 