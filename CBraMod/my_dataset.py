import os
import mne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class MI2aDataset(Dataset):
    def __init__(self, gdf_file_path):
        print(f"正在加载数据: {gdf_file_path}")
        # 1. 读取 GDF 文件
        raw = mne.io.read_raw_gdf(gdf_file_path, preload=True, verbose=False)

        # 2. 提取事件 (769:左手, 770:右手, 771:双脚, 772:舌头)
        event_id = {'769': 0, '770': 1, '771': 2, '772': 3}
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)

        # 3. 切分 Epochs
        tmin, tmax = 0.5, 3.7
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                            baseline=None, preload=True, verbose=False,
                            event_repeated='drop')

        # 4. 获取原始数据并强制对齐维度
        self.data = epochs.get_data()
        self.data = self.data[:, :22, :]  # 只取前 22 个 EEG 通道
        self.data = self.data[:, :, :800]  # 只取前 800 个时间点

        # 5. 获取标签
        self.labels = epochs.events[:, 2]

        # 6. 重塑数据形状以适配 CBraMod: (N, 22, 4, 200)
        self.data = self.data.reshape(-1, 22, 4, 200)
        print(f"✅ 数据加载完成！共提取 {len(self.data)} 个样本，最终形状: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])


def get_mi2a_dataloader(gdf_file_path, batch_size=8, num_workers=0):
    dataset = MI2aDataset(gdf_file_path)

    # ================= 核心修改：划分训练、验证、测试集 =================
    # 按照 70% 训练，15% 验证，15% 测试的比例随机划分
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 生成三个 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 返回包含三个键的字典，完美契合原作者 Trainer 的要求！
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}