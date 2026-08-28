"""Subject-independent SHIN Dataset B training entry for fNIRS-T."""

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import fNIRS_T  # noqa: E402
from shin_pipeline.data import load_subjects  # noqa: E402


def parse_subjects(text):
    result = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not result or len(result) != len(set(result)) or any(x < 1 or x > 29 for x in result):
        raise ValueError(f"invalid SHIN subject list: {text!r}")
    return result


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target[:, None]).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * smooth).mean()


def make_loader(x, y, batch_size, shuffle):
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model, loader, device, criterion):
    model.eval()
    losses, logits_all, labels_all = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            losses.append(float(criterion(logits, y).item()) * len(y))
            logits_all.append(logits.cpu())
            labels_all.append(y.cpu())
    logits = torch.cat(logits_all).numpy()
    labels = torch.cat(labels_all).numpy()
    predictions = logits.argmax(axis=1)
    return {
        "loss": float(sum(losses) / len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(labels, predictions)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).astype(int).tolist(),
    }


def write_experiment_record(out_dir, summary, splits):
    note = str(summary["experiment_note"]).replace("|", "\\|")
    best, final = summary["best_test"], summary["final_test"]
    lines = [
        "# 实验记录",
        "",
        "## 实验思路",
        "",
        note,
        "",
        "## 实验参数",
        "",
        "| 参数 | 取值 |",
        "|---|---|",
        "| 数据集 | SHIN fNIRS / Dataset B |",
        f"| 模型 | {summary['model']} |",
        f"| 随机种子 | {summary['seed']} |",
        f"| 训练轮数 | {summary['epochs']} |",
        f"| 分类头学习率 | {summary['head_lr']} |",
        f"| 骨干网络学习率 | {summary['backbone_lr']} |",
        f"| 训练集受试者 | {','.join(map(str, splits['train']))} |",
        f"| 验证集受试者 | {','.join(map(str, splits['val']))} |",
        f"| 测试集受试者 | {','.join(map(str, splits['test']))} |",
        "",
        "## 实验结果",
        "",
        "| 检查点 | 轮次 | 验证准确率 | 测试准确率 | Macro F1 | Kappa |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 验证集最佳 | {summary['best_epoch']} | {summary['best_val_accuracy']:.4f} | {best['accuracy']:.4f} | {best['f1_macro']:.4f} | {best['kappa']:.4f} |",
        f"| 最后一轮 | {summary['epochs']} | {summary['history'][-1]['val']['accuracy']:.4f} | {final['accuracy']:.4f} | {final['f1_macro']:.4f} | {final['kappa']:.4f} |",
        "",
        f"最佳模型测试集混淆矩阵：`{best['confusion_matrix']}`",
        "",
    ]
    (out_dir / "EXPERIMENT_RECORD.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train fNIRS-T on SHIN Dataset B")
    parser.add_argument("--data-root", default=r"D:\DataSets\SHIN\NIRS_01-29")
    parser.add_argument("--task", choices=("mi", "ma"), default="ma")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--train-subjects", default=",".join(str(i) for i in range(1, 20)))
    parser.add_argument("--val-subjects", default="20,21,22,23,24")
    parser.add_argument("--test-subjects", default="25,26,27,28,29")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--experiment-note",
        default="使用论文发布的 fNIRS-T 架构和直接适配的 SHIN fNIRS 数据，评估心算与静息任务的跨受试者分类性能。",
    )
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    splits = {
        "train": parse_subjects(args.train_subjects),
        "val": parse_subjects(args.val_subjects),
        "test": parse_subjects(args.test_subjects),
    }
    all_subjects = [value for values in splits.values() for value in values]
    if len(all_subjects) != len(set(all_subjects)):
        raise ValueError("train/val/test subject lists must be disjoint")
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else Path(r"D:\data\fNIRS-Transformer-SHIN") / (
        datetime.now().strftime("%Y%m%d-%H%M%S")
        + f"_ep{args.epochs}_headlr{args.head_lr:g}_backbonelr{args.backbone_lr:g}_seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    arrays, labels, infos = {}, {}, {}
    for name, subjects in splits.items():
        arrays[name], labels[name], infos[name] = load_subjects(
            data_root, subjects, task=args.task
        )
        print(f"{name}: x={arrays[name].shape} labels={dict(Counter(labels[name].tolist()))}")
    diagnostics = {
        "dataset": "SHIN / original paper Dataset B",
        "task": args.task,
        "data_root": str(data_root),
        "splits": splits,
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
        "label_map": (
            {"left_hand": 0, "right_hand": 1}
            if args.task == "mi"
            else {"mental_arithmetic": 0, "baseline_rest": 1}
        ),
        "normalization": "per-trial global z-score",
        "seed": args.seed,
        "experiment_note": args.experiment_note,
        "subjects": infos,
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    if args.diagnose_only:
        print(f"diagnostics written to {out_dir / 'diagnostics.json'}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fNIRS_T(
        n_class=2, sampling_point=200, dim=64, depth=6, heads=8, mlp_dim=64
    ).to(device)
    head_parameters = list(model.mlp_head.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = LabelSmoothing(0.1)
    loaders = {
        name: make_loader(arrays[name], labels[name], args.batch_size, name == "train")
        for name in arrays
    }

    best_accuracy, best_epoch, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_count = 0.0, 0, 0
        flooding_level = 0.45 if epoch <= 30 else 0.40 if epoch <= 50 else 0.35
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            raw_loss = criterion(logits, y)
            loss = (raw_loss - flooding_level).abs() + flooding_level
            loss.backward()
            optimizer.step()
            train_loss += float(raw_loss.item()) * len(y)
            train_correct += int((logits.argmax(dim=1) == y).sum().item())
            train_count += len(y)
        val = evaluate(model, loaders["val"], device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / train_count,
            "train_accuracy": train_correct / train_count,
            "val": val,
            "head_lr": optimizer.param_groups[1]["lr"],
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "flooding_level": flooding_level,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} val_acc={val['accuracy']:.4f}"
        )
        if val["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = val["accuracy"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val}, out_dir / "best.pt")
        scheduler.step()

    final_test = evaluate(model, loaders["test"], device, criterion)
    torch.save(
        {"model": model.state_dict(), "epoch": args.epochs, "test": final_test}, out_dir / "last.pt"
    )
    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    best_test = evaluate(model, loaders["test"], device, criterion)
    summary = {
        "device": str(device),
        "seed": args.seed,
        "task": args.task,
        "experiment_note": args.experiment_note,
        "model": "fNIRS-T Dataset B (dim=64, depth=6, heads=8, mlp_dim=64)",
        "epochs": args.epochs,
        "head_lr": args.head_lr,
        "backbone_lr": args.backbone_lr,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "best_test": best_test,
        "final_test": final_test,
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_experiment_record(out_dir, summary, splits)
    print(json.dumps({key: summary[key] for key in ("device", "best_epoch", "best_val_accuracy", "best_test", "final_test")}, indent=2))


if __name__ == "__main__":
    main()
