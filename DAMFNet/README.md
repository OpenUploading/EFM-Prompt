# DAMFNet: Dual-Attention Multi-Node EEG–fNIRS Fusion Network for Motor Imagery Classification

This repository provides an end-to-end implementation of **DAMFNet**, a hybrid **EEG–fNIRS** fusion framework for motor imagery (MI) classification. DAMFNet adopts an asynchronous dual-branch backbone (EEG branch and fNIRS branch) and performs **dual-node fusion** (spatial fusion node and temporal fusion node) with **spatial/temporal attention** to learn complementary cross-modal representations.

---

## 1. Environment

- Python >= 3.8 (tested with Python 3.10)
- PyTorch >= 1.10
- NumPy
- scikit-learn
- matplotlib (optional; only used by visualization utilities)
- pandas (optional; only for CSV convenience)

Install dependencies:

```bash
pip install numpy scikit-learn matplotlib pandas
```

---

## 2. Repository Structure

A typical layout:

```text
DAMFNet/
  model/                       # network definitions
    eeg_net.py
    Hbr_net.py
    ST_Attention.py
    fusion_net.py              # DAMFNet fusion model (core)
  DataProcess.py               # data loading + sliding window
  training.py                  # unimodal training utilities
  fusion_training.py           # fusion training utilities
  func.py / func4.py           # metrics / batching / optional plotting
  main.py                      # baseline/unimodal entry (Dataset1 by default)
  data/
    Raw_data/                  # Dataset1 (public)
    Mydata/                    # Dataset2 (private)
```

## 3. Data Organization (Required)

### 3.1 Dataset1 (Public): `data/Raw_data/`

```text
data/Raw_data/
  EEG/
    X_eeg{subject_id}.npy
    Y_eeg{subject_id}.npy
  HbO/
    X_hbo{subject_id}.npy
    Y_hbo{subject_id}.npy
  HbR/
    X_hbr{subject_id}.npy
    Y_hbr{subject_id}.npy
```

### 3.2 Dataset2 (Private): `data/Mydata/`

```text
data/Mydata/
  EEG/
    X_eeg{subject_id}.npy
    Y_eeg{subject_id}.npy
  fNIRS/
    X_hbo{subject_id}.npy
    X_hbr{subject_id}.npy
    Y_nirs{subject_id}.npy
```

Dataset2 labels may need mapping (e.g., `{4,5,6,7} -> {0,1,2,3}`), see `hbo_main_dataset2.py`.

---

## 4. Evaluation Protocol (Reproducibility)

Within-subject evaluation uses **10-fold cross-validation** per subject.

For each fold:
- the data are split into train/test
- validation is split from training only (for model selection and early stopping)
- the fold test set is not used for training or model selection

---

