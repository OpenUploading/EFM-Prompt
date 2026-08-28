"""Run portable EEGNet on cross-subject FineMI event 1 versus event 6."""

from collections import Counter
from pathlib import Path

import numpy as np

import run_shin_eegnet as portable


def load_split(root, name, subjects, cache_dir, task_key, task):
    arrays, labels, details = [], [], []
    for subject in subjects:
        path = Path(root) / f"subject{subject:02d}_paired.npz"
        with np.load(path, allow_pickle=False) as item:
            arrays.append(item["eeg"].astype(np.float32))
            labels.append(item["labels"].astype(np.int64))
        details.append({"subject": subject, "source": str(path)})
    x_all, y_all = np.concatenate(arrays), np.concatenate(labels)
    print(f"[{name}] FineMI binary: X={x_all.shape}, y={Counter(y_all.tolist())}", flush=True)
    return x_all, y_all, details


def record(summary):
    best, test, final = summary["best"], summary["best_test"], summary["final"]
    return f"""# EEGNet-8,2 × FineMI 跨被试二分类

| 参数 | 值 |
|---|---|
| 任务 | 类别1手部开合 vs 类别6肩关节旋转 |
| 划分 | train 1–12 / val 13–15 / test 16–18 |
| 输入 | 62×800，200 Hz，0–4秒；CAR＋逐trial逐通道z-score |
| 最佳epoch | {best['epoch']} |
| 最佳Val Acc | {best['val']['accuracy']:.4f} |
| 对应Test Acc | {test['accuracy']:.4f} |
| Test Macro-F1 | {test['f1_macro']:.4f} |
| Test Kappa | {test['cohen_kappa']:.4f} |
| 最终Test Acc | {final['test']['accuracy']:.4f} |
"""


if __name__ == "__main__":
    portable.TASKS["mi"] = {
        "name": "FineMI-binary-1v6",
        "description": "hand open/close (0) vs shoulder pronation/supination (1)",
        "sessions": (),
        "labels": {"hand_open_close": 0, "shoulder_pron_sup": 1},
    }
    portable.load_split = load_split
    portable.record = record
    portable.main()
