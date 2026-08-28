import glob
import os

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import butter, lfilter, resample
from torch.utils.data import DataLoader, Dataset, random_split

def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

VAL_RATIO_FROM_TEST = 0.5


class CustomDataset(Dataset):
    def __init__(self, data_dir, mode='train', sfreq=250, trial_duration=4.0):
        self.mode = mode
        self.sfreq = sfreq
        self.root_dir = self._resolve_mat_root(data_dir)
        self.file_paths = self._resolve_mat_files(mode)
        self.data = []
        self.labels = []
        self._load_data()

    def _resolve_mat_root(self, data_dir):
        candidates = [
            os.path.join(data_dir, 'MI_BCIC_IV2a_mat'),
            os.path.join(data_dir, 'data_mat'),
            data_dir,
        ]
        for candidate in candidates:
            if os.path.exists(os.path.join(candidate, 'A01T.mat')):
                return candidate
        raise FileNotFoundError(
            f"Cannot find BCIC IV 2a mat files under {data_dir}. "
            f"Expected extracted files like A01T.mat in MI_BCIC_IV2a_mat or data_mat."
        )

    def _resolve_mat_files(self, mode):
        suffix = 'T.mat' if mode == 'train' else 'E.mat'
        file_paths = sorted(glob.glob(os.path.join(self.root_dir, f"*{suffix}")))
        if not file_paths:
            raise FileNotFoundError(f"No {suffix} files found under {self.root_dir}")
        return file_paths

    def _preprocess_trial(self, trial_data):
        """按项目原始 BCIC-IV-2a 预处理输出 (22, 4, 200)"""
        sample = trial_data[:, :22].transpose(1, 0).astype(np.float32)
        sample = sample - np.mean(sample, axis=0, keepdims=True)
        b, a = butter_bandpass(0.3, 50, self.sfreq)
        sample = lfilter(b, a, sample, axis=-1)
        sample = sample[:, 2 * self.sfreq:6 * self.sfreq]
        sample = resample(sample, 800, axis=-1)
        if sample.shape != (22, 800):
            raise ValueError(f"Unexpected BCIC sample shape after preprocess: {sample.shape}")
        return sample.reshape(22, 4, 200).astype(np.float32)

    def _load_data(self):
        print(f"\n--- Loading {self.mode} data from {len(self.file_paths)} .mat files ---")
        total_trials = 0

        for file_path in self.file_paths:
            file_name = os.path.basename(file_path)
            try:
                data = loadmat(file_path)
                num_sessions = len(data['data'][0])
                for j in range(3, num_sessions):
                    raw_data = data['data'][0, j][0, 0][0]
                    events = data['data'][0, j][0, 0][1][:, 0]
                    labels = data['data'][0, j][0, 0][2][:, 0]
                    signal_length = raw_data.shape[0]
                    event_positions = events.tolist()
                    event_positions.append(signal_length)
                    trial_ranges = [
                        (event_positions[i], event_positions[i + 1])
                        for i in range(len(event_positions) - 1)
                    ]

                    for (start, end), label in zip(trial_ranges, labels):
                        sample = self._preprocess_trial(raw_data[start:end])
                        self.data.append(sample)
                        self.labels.append(int(label) - 1)
                        total_trials += 1
            except Exception as e:
                print(f"  [Warning] Failed to load {file_name}. Error: {e}")

        print(f"Successfully loaded {total_trials} trials for {self.mode} mode.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class LoadDataset:
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        if len(train_set) == 0:
            raise ValueError("Training set is empty! Please check the BCIC IV 2a T.mat files.")

        full_test_set = CustomDataset(self.datasets_dir, mode='test')
        if len(full_test_set) == 0:
            raise ValueError("Test set is empty! Please check the BCIC IV 2a E.mat files.")

        val_size = int(len(full_test_set) * VAL_RATIO_FROM_TEST)
        test_size = len(full_test_set) - val_size
        if val_size == 0 or test_size == 0:
            raise ValueError("The official test split is too small to be divided into val/test.")

        generator = torch.Generator().manual_seed(self.params.seed)
        val_set, test_set = random_split(full_test_set, [val_size, test_size], generator=generator)
        print(f"Split official test set into val/test: {len(val_set)}/{len(test_set)}")

        train_loader = DataLoader(train_set, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, pin_memory=True)

        return {'train': train_loader, 'val': val_loader, 'test': test_loader}
