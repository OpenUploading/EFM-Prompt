"""Run MI/MA MoPE component ablations in matched GPU pairs."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_cbramod_mope_boundary_gpu.ps1"
LOG_ROOT = ROOT / "batch_logs" / "mope_component_ablation"
LOG_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ROOT = ROOT / "runs_mope"
HEADS = {
    "mi": ROOT / "runs" / "cbramod_boundary_prompt_mi_conditional_joint_eeg_only_aligned_ep50_featurelr0.0003_headlr0.0001_m4_r8_seed1_20260801-145401" / "best_model.pth",
    "ma": ROOT / "runs" / "cbramod_boundary_prompt_ma_conditional_joint_eeg_only_aligned_ep50_featurelr0.0003_headlr0.0001_m4_r8_seed1_20260801-145417" / "best_model.pth",
}


def is_complete(strategy: str, dropped: str, task: str) -> bool:
    pattern = (
        f"cbramod_mope_boundary_{task}_conditional_{strategy}_"
        f"pre_post_aligned_drop{dropped}_*"
    )
    return any((run_dir / "summary.json").is_file() for run_dir in RUN_ROOT.glob(pattern))


def main() -> None:
    for strategy in ("prompt_only", "joint"):
        for dropped in ("static", "dynamic", "mapped"):
            processes = []
            handles = []
            for task in ("mi", "ma"):
                if is_complete(strategy, dropped, task):
                    print(f"SKIP complete strategy={strategy} drop={dropped} task={task}", flush=True)
                    continue
                tag = f"{strategy}_{dropped}_{task}"
                stdout = (LOG_ROOT / f"{tag}.out.log").open("w", encoding="utf-8")
                stderr = (LOG_ROOT / f"{tag}.err.log").open("w", encoding="utf-8")
                command = [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(RUNNER),
                    "-Task", task, "-Mode", "pre_post", "-PromptSource", "conditional",
                    "-TrainingStrategy", strategy, "-DropComponent", dropped, "-Epochs", "50",
                ]
                if strategy == "prompt_only":
                    command.extend(("-HeadCheckpoint", str(HEADS[task])))
                processes.append((task, subprocess.Popen(command, stdout=stdout, stderr=stderr)))
                handles.extend((stdout, stderr))
            failures = []
            for task, process in processes:
                return_code = process.wait()
                if return_code != 0:
                    failures.append((task, return_code))
            for handle in handles:
                handle.close()
            if failures:
                raise RuntimeError(f"Ablation failed: strategy={strategy}, drop={dropped}, {failures}")
            print(f"COMPLETE strategy={strategy} drop={dropped} tasks=mi,ma", flush=True)


if __name__ == "__main__":
    main()
