"""Run portable EEGNet-8,2 on FineMI's strict 12/3/3 subject split."""

from collections import Counter
from pathlib import Path

import numpy as np

import run_shin_eegnet as portable


DEFAULT_CACHE = Path(
    r"D:\0senior student creation\datasets\FineMI_Yi2025_raw\processed_200hz_no_car"
)


def load_split(root, name, subjects, cache_dir, task_key, task):
    arrays, labels, details = [], [], []
    for subject in subjects:
        path = Path(root) / f"subject{subject:02d}_paired.npz"
        with np.load(path, allow_pickle=False) as item:
            x = item["eeg"].astype(np.float32)
            y = item["labels"].astype(np.int64)
        arrays.append(x)
        labels.append(y)
        details.append({"subject": subject, "source": str(path)})
    x_all = np.concatenate(arrays)
    y_all = np.concatenate(labels)
    print(f"[{name}] FineMI: X={x_all.shape}, y={Counter(y_all.tolist())}", flush=True)
    return x_all, y_all, details


def record(summary):
    """Write a FineMI-specific record instead of SHIN's fixed 19/5/5 text."""
    best, test, final = summary["best"], summary["best_test"], summary["final"]
    return f"""# EEGNet-8,2 × FineMI EEG 实验记录

## 参数

| 参数 | 值 |
|---|---|
| 任务 | {summary['task']['name']}：{summary['task']['description']} |
| 划分 | 跨被试 12/3/3：train 1–12 / val 13–15 / test 16–18 |
| 输入 | 62×800，200 Hz，0–4 秒，逐 trial 全局 z-score；保留原始 M1 参考，不额外 CAR |
| 模型 | EEGNet-8,2，F1=8，D=2，F2=16 |
| 参数量 | {summary['model']['parameters']} |
| Seed | {summary['seed']} |
| Epoch | {summary['schedule']['epochs']} |
| Batch size | {summary['schedule']['batch_size']} |
| 学习率 | {summary['schedule']['lr']} |
| Weight decay | {summary['schedule']['weight_decay']} |
| Dropout | {summary['schedule']['dropout']} |

## 结果

| 检查点 | Epoch | Val Acc | Test Acc | Test Macro-F1 | Test Kappa |
|---|---:|---:|---:|---:|---:|
| 最佳验证模型 | {best['epoch']} | {best['val']['accuracy']:.4f} | {test['accuracy']:.4f} | {test['f1_macro']:.4f} | {test['cohen_kappa']:.4f} |
| 最终模型 | {final['epoch']} | {final['val']['accuracy']:.4f} | {final['test']['accuracy']:.4f} | {final['test']['f1_macro']:.4f} | {final['test']['cohen_kappa']:.4f} |
"""


if __name__ == "__main__":
    portable.TASKS["mi"] = {
        "name": "FineMI-8class-MI",
        "description": "8 unilateral upper-limb joint motor-imagery classes",
        "sessions": (),
        "labels": {
            "hand_open_close": 0,
            "wrist_flex_ext": 1,
            "wrist_abd_add": 2,
            "elbow_pron_sup": 3,
            "elbow_flex_ext": 4,
            "shoulder_pron_sup": 5,
            "shoulder_abd_add": 6,
            "shoulder_flex_ext": 7,
        },
    }
    portable.load_split = load_split
    portable.record = record
    portable.main()
