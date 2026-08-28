"""Deterministic subject-wise stratified 5-fold assignment for HEFMI-ICH."""

from __future__ import annotations

from collections import Counter

import numpy as np


SPLIT_TO_FOLDS = {
    "train": (1, 2, 3),
    "val": (4,),
    "test": (5,),
}


def make_within_subject_five_fold_indices(
    labels: np.ndarray,
    subjects: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Assign every subject's trials to stratified folds, then map 3/1/1.

    Fold assignment is independently shuffled for each subject and class using
    ``seed + subject * 10007``.  This is one fixed split derived from five
    folds, not five rotating cross-validation runs.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    subjects = np.asarray(subjects, dtype=np.int64).reshape(-1)
    if labels.shape != subjects.shape or labels.size == 0:
        raise ValueError(
            f"labels/subjects must be non-empty aligned vectors, got "
            f"{labels.shape} and {subjects.shape}"
        )

    split_parts: dict[str, list[np.ndarray]] = {name: [] for name in SPLIT_TO_FOLDS}
    subject_rows: list[dict] = []
    for subject in sorted(np.unique(subjects).tolist()):
        subject_indices = np.flatnonzero(subjects == subject)
        subject_labels = labels[subject_indices]
        fold_parts: list[list[np.ndarray]] = [[] for _ in range(5)]
        rng = np.random.default_rng(int(seed) + int(subject) * 10007)
        for label in sorted(np.unique(subject_labels).tolist()):
            class_indices = subject_indices[subject_labels == label]
            if class_indices.size < 5:
                raise ValueError(
                    f"sub-{subject} class {label} has only {class_indices.size} "
                    "trials; stratified 5-fold requires at least 5"
                )
            class_indices = rng.permutation(class_indices)
            for fold_index, chunk in enumerate(np.array_split(class_indices, 5)):
                fold_parts[fold_index].append(chunk)

        folds = [
            np.sort(np.concatenate(parts)).astype(np.int64, copy=False)
            for parts in fold_parts
        ]
        fold_counts = []
        for fold_index, fold_indices in enumerate(folds, 1):
            fold_counts.append({
                "fold": fold_index,
                "trials": int(fold_indices.size),
                "label_counts": {
                    str(key): int(value)
                    for key, value in sorted(Counter(labels[fold_indices].tolist()).items())
                },
            })
        subject_rows.append({
            "subject": int(subject),
            "total_trials": int(subject_indices.size),
            "folds": fold_counts,
        })
        for split_name, fold_numbers in SPLIT_TO_FOLDS.items():
            chosen = np.sort(np.concatenate([folds[number - 1] for number in fold_numbers]))
            split_parts[split_name].append(chosen)

    split_indices = {
        name: np.sort(np.concatenate(parts)).astype(np.int64, copy=False)
        for name, parts in split_parts.items()
    }
    concatenated = np.concatenate(list(split_indices.values()))
    if (
        concatenated.size != labels.size
        or np.unique(concatenated).size != labels.size
        or int(concatenated.min()) != 0
        or int(concatenated.max()) != labels.size - 1
    ):
        raise RuntimeError("within-subject fold assignment is not exhaustive/disjoint")

    metadata = {
        "protocol": "within_subject_stratified_5fold_fixed_3_1_1",
        "description": (
            "Each subject is independently stratified into five folds; "
            "folds 1-3 train, fold 4 validation, fold 5 test."
        ),
        "seed": int(seed),
        "folds": {name: list(values) for name, values in SPLIT_TO_FOLDS.items()},
        "splits": {
            name: {
                "trials": int(indices.size),
                "subjects": sorted(np.unique(subjects[indices]).astype(int).tolist()),
                "label_counts": {
                    str(key): int(value)
                    for key, value in sorted(Counter(labels[indices].tolist()).items())
                },
            }
            for name, indices in split_indices.items()
        },
        "subject_folds": subject_rows,
    }
    return split_indices, metadata
