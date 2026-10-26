"""Run, resume, and summarize leave-one-subject-out experiments."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from . import train as trainer
from .config import RunConfig, resolve_paths
from .paths import workspace_path
from .train import run_training


def resume_config_differences(previous, subject_config: RunConfig, allow_schedule_change=False):
    """Compare experiment settings after resolving paths from saved configurations."""
    subject_config = resolve_paths(subject_config)
    expected = trainer._jsonable(asdict(subject_config))
    previous = dict(previous)
    for group, keys in (("data", ("dataset_root", "visual_feature_dir")),
                        ("visual_lora", ("checkpoint",))):
        previous[group] = dict(previous.get(group, {}))
        for key in keys:
            if previous[group].get(key) is not None:
                previous[group][key] = str(workspace_path(previous[group][key]))
    differences = []
    for field in ("data", "model", "optimization", "visual_lora", "score", "run_name"):
        old = previous.get(field)
        new = expected[field]
        if field == "optimization" and isinstance(old, dict):
            old = dict(old)
            new = dict(new)
            old.pop("epochs", None)
            new.pop("epochs", None)
            if allow_schedule_change:
                for key in ("coarse_epochs", "learning_rate", "fine_learning_rate"):
                    old.pop(key, None)
                    new.pop(key, None)
            if subject_config.optimization.loss_mode == "samga":
                for key in ("loss_mode", "score_alignment_scale", "score_recovery_weight", "score_recovery_temperature", "score_recovery_identity_ratio", "score_recovery_csls_neighbors", "score_recovery_landmark_schedule"):
                    old.setdefault(key, new[key])
            else:
                old.setdefault("paired_loss_version", new["paired_loss_version"])
        if field == "visual_lora" and allow_schedule_change and isinstance(old, dict):
            old = dict(old)
            new = dict(new)
            for key in ("learning_rate", "coarse_learning_rate", "stage"):
                old.pop(key, None)
                new.pop(key, None)
        if isinstance(old, dict) and isinstance(new, dict):
            differences.extend(f"{field}.{key}" for key in sorted(old.keys() | new.keys()) if old.get(key) != new.get(key))
        elif old != new:
            differences.append(field)
    if not allow_schedule_change and previous.get("coarse_phase_start_epoch", 1) != subject_config.coarse_phase_start_epoch:
        differences.append("coarse_phase_start_epoch")
    return differences


def prior_run(subject_config: RunConfig):
    """Find the most advanced readable checkpoint with compatible settings."""
    matches = []
    for folder in subject_config.checkpoint_root.glob(subject_config.run_name + "_*"):
        config_path = folder / "config.json"
        last_path = folder / "last.pt"
        if not config_path.is_file() or not last_path.is_file():
            continue
        try:
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if resume_config_differences(previous, subject_config):
                continue
            payload = torch.load(last_path, map_location="cpu", weights_only=False, mmap=True)
            last_epoch = int(payload["epoch"])
            if (subject_config.visual_lora.enabled
                    and last_epoch > subject_config.optimization.coarse_epochs
                    and not payload.get("visual_lora_state_dict")):
                print(f"Ignoring checkpoint without trained LoRA weights: {last_path}", flush=True)
                continue
            matches.append((last_epoch, last_path.stat().st_mtime, folder))
            del payload
        except (OSError, RuntimeError, EOFError, ValueError, KeyError) as exc:
            print(f"Ignoring unreadable checkpoint {last_path}: {exc}", flush=True)
    return max(matches, default=None)


def run_loso(config: RunConfig, held_out_subjects: tuple[int, ...], run_name_prefix: str,
             total_epochs: int, coarse_epochs: int) -> dict:
    """Train or resume requested folds; an explicit checkpoint starts an extended schedule."""
    loss_mode = config.optimization.loss_mode
    config = resolve_paths(config)
    summary_path = config.checkpoint_root / ("loso_summary.json" if loss_mode == "samga" else "loso_score_loss_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}


    for held_out_subject in held_out_subjects:
        subject_config = replace(
            config,
            data=replace(config.data, held_out_subject=held_out_subject),
            run_name=f"{run_name_prefix}_subject_{held_out_subject:02d}",
            resume_checkpoint=None,
        )
        explicit_resume = config.resume_checkpoint if held_out_subject == config.data.held_out_subject else None
        if explicit_resume is not None:
            explicit_resume = Path(explicit_resume)
            if not explicit_resume.is_file():
                raise FileNotFoundError(f"Resume checkpoint does not exist: {explicit_resume}")
            saved_config_path = explicit_resume.parent / "config.json"
            if not saved_config_path.is_file():
                raise FileNotFoundError(f"Cannot verify resume settings: {saved_config_path}")
            saved_config = json.loads(saved_config_path.read_text(encoding="utf-8"))
            differences = resume_config_differences(saved_config, subject_config, allow_schedule_change=True)
            if differences:
                raise ValueError("Checkpoint model/data settings differ: " + ", ".join(differences))
            payload = torch.load(explicit_resume, map_location="cpu", weights_only=False, mmap=True)
            last_epoch = int(payload["epoch"])
            del payload
            saved_coarse_epochs = int(saved_config["optimization"]["coarse_epochs"])
            extended_total_epochs = last_epoch + total_epochs
            extended_coarse_epochs = last_epoch + coarse_epochs if coarse_epochs else min(last_epoch, saved_coarse_epochs)
            if extended_coarse_epochs >= extended_total_epochs:
                raise ValueError("TOTAL_EPOCHS must leave at least one fine epoch after the added coarse epochs")
            new_coarse_cycle = last_epoch > saved_coarse_epochs and coarse_epochs > 0
            branch_name = f"{subject_config.run_name}_from_{explicit_resume.parent.name}_{explicit_resume.stem}_e{last_epoch}"
            subject_config = replace(
                subject_config,
                run_name=branch_name,
                optimization=replace(subject_config.optimization, epochs=extended_total_epochs,
                                     coarse_epochs=extended_coarse_epochs),
                coarse_phase_start_epoch=last_epoch + 1 if new_coarse_cycle else 1,
            )
            previous = prior_run(subject_config)
            if previous is not None and previous[0] >= extended_total_epochs:
                print(f"Subject {held_out_subject:02d}: extended run already complete in {previous[2]}", flush=True)
                continue
            resume_path = previous[2] / "last.pt" if previous is not None else explicit_resume
            resume_epoch = previous[0] if previous is not None else last_epoch
            subject_config = replace(subject_config, resume_checkpoint=resume_path)
            print(f"Subject {held_out_subject:02d}: continuing from epoch {resume_epoch}; coarse through {extended_coarse_epochs}, total through {extended_total_epochs}", flush=True)
        else:
            previous = prior_run(subject_config)
            if previous is not None:
                last_epoch, _, folder = previous
                if last_epoch >= subject_config.optimization.epochs:
                    print(f"Subject {held_out_subject:02d}: already complete in {folder}", flush=True)
                    continue
                subject_config = replace(subject_config, resume_checkpoint=folder / "last.pt")
                print(f"Subject {held_out_subject:02d}: resuming after epoch {last_epoch}", flush=True)
            else:
                print(f"Subject {held_out_subject:02d}: starting from epoch 1", flush=True)
        run_directory = run_training(subject_config)
        history = json.loads((run_directory / "history.json").read_text(encoding="utf-8"))
        final = history[-1]
        best = max(history, key=lambda row: row["validation_test_r1"])
        summary[str(held_out_subject)] = {
            "held_out_subject": held_out_subject,
            "train_subjects": [subject for subject in range(1, 11) if subject != held_out_subject],
            "run_directory": str(run_directory.resolve()),
            "last_epoch": int(final["epoch"]),
            "final_r1": float(final["validation_test_r1"]),
            "final_r5": float(final["validation_test_r5"]),
            "final_ic": float(final["validation_test_ic"]),
            "best_r1_epoch": int(best["epoch"]),
            "best_r1": float(best["validation_test_r1"]),
            "r5_at_best_r1": float(best["validation_test_r5"]),
        }
        if config.score.enabled:
            for method in ("uncalibrated", "random", "csls_selected"):
                for ranking in ("cosine", "csls"):
                    prefix = f"validation_score_{method}_{ranking}"
                    summary[str(held_out_subject)][f"final_{method}_{ranking}_r1"] = float(final[f"{prefix}_r1"])
                    summary[str(held_out_subject)][f"final_{method}_{ranking}_r5"] = float(final[f"{prefix}_r5"])
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Subject {held_out_subject:02d}: final R@1={final['validation_test_r1']:.4f} "
              f"R@5={final['validation_test_r5']:.4f}", flush=True)

    completed = [summary[str(subject)] for subject in held_out_subjects if str(subject) in summary]
    if completed and config.score.enabled:
        aggregate = {"completed_subjects": len(completed), "subjects": [row["held_out_subject"] for row in completed]}
        for method in ("uncalibrated", "random", "csls_selected"):
            for ranking in ("cosine", "csls"):
                for metric in ("r1", "r5"):
                    name = f"final_{method}_{ranking}_{metric}"
                    aggregate[name] = sum(row[name] for row in completed) / len(completed)
                print(f"{method:14s} {ranking:6s} mean R@1={aggregate[f'final_{method}_{ranking}_r1']:.4f} "
                      f"R@5={aggregate[f'final_{method}_{ranking}_r5']:.4f} "
                      f"over {len(completed)} subjects")
        aggregate_path = config.checkpoint_root / ("loso_score_aggregate.json" if loss_mode == "samga" else "loso_score_loss_aggregate.json")
        aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"Aggregate: {aggregate_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    return summary
