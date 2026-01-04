import os
# 在导入 kagglehub 之前设置下载根目录
os.environ["KAGGLEHUB_CACHE"] = "/home/wangzh/code-sapce/NLPrompt/data/food101N"

import kagglehub

# 执行下载
# 注意：kagglehub 会在该路径下自动创建 datasets/kuanghueilee/food-101n/versions/x 结构
path = kagglehub.dataset_download("kuanghueilee/food-101n")

print("数据集已下载至:", path)