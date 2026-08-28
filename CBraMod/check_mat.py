from scipy.io import loadmat
import numpy as np

# 检查 .mat 文件内容
mat_file = r'D:\BaiduNetdiskDownload\MI_BCI_IV_2a\true_labels\A01E.mat'
data = loadmat(mat_file)

print("Keys in .mat file:", list(data.keys()))
print()
for key in data.keys():
    if not key.startswith('__'):
        val = data[key]
        print(f"{key}:")
        print(f"  Type: {type(val)}")
        if hasattr(val, 'shape'):
            print(f"  Shape: {val.shape}")
        if hasattr(val, 'flatten'):
            unique_vals = np.unique(val)
            print(f"  Unique values: {unique_vals}")
            print(f"  First 10 values: {val.flatten()[:10]}")
        print()
