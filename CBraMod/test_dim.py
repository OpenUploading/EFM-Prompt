import sys
sys.path.insert(0, r'D:\CBraMod-main\CBraMod-main')
import os
os.chdir(r'D:\CBraMod-main\CBraMod-main')

# 禁用 MNE 警告
import warnings
warnings.filterwarnings('ignore')

from datasets.bciciv2a_dataset import CustomDataset

# 测试数据集
dataset = CustomDataset(r'D:\BaiduNetdiskDownload\MI_BCI_IV_2a', mode='train')
print(f"Dataset size: {len(dataset)}")

if len(dataset) > 0:
    x, y = dataset[0]
    print(f"Sample shape: {x.shape}")
    print(f"Label: {y}")
    
    # 保存到文件
    with open('test_output.txt', 'w') as f:
        f.write(f"Dataset size: {len(dataset)}\n")
        f.write(f"Sample shape: {x.shape}\n")
        f.write(f"Label: {y}\n")
else:
    print("Dataset is empty!")
    with open('test_output.txt', 'w') as f:
        f.write("Dataset is empty!\n")
