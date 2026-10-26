"""Inspect saved retrieval histories or evaluate labeled-support calibration.

Use --history-only to avoid inference. Otherwise requested scores are cached
next to checkpoints; encoder weights and training histories stay unchanged."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

from things_eeg2.paths import workspace_path
import queue
import re
from statistics import mean
import threading
from typing import Any

from plot_checkpoint_dynamics import _selected_paths


CHECKPOINT_ROOT = workspace_path("checkpoints/things_eeg2_samga_loso_rank192")
USE_ALL_CHECKPOINTS = True
ALL_CHECKPOINT_KIND = "both"
SORT_PAGES_BY_DATE = True
SELECTED_CHECKPOINTS: tuple[Path, ...] = (

)


SCORE_POOL_CLASSES = 241
SCORE_LANDMARKS = 240
SCORE_SEED = 42
SCORE_CSLS_NEIGHBORS = 100
SCORE_RHO = 0.125


SCORE_MAP_STAGES: tuple[tuple[str, int, float], ...] = (
    ("csls_selected", 120, 0.25),
    ("csls_selected", 120, 0.25),
    ("csls_selected", 120, 0.25),
    ("csls_selected", 120, 0.25),
)
AUTO_EVALUATE_ALL = True

SCHEMA_VERSION = 1
RANKINGS = ("cosine", "csls")
BASE_METHODS = ("uncalibrated", "random", "csls_selected")
CHAIN_METHODS = tuple(f"chain_{index}" for index in range(1, len(SCORE_MAP_STAGES) + 1))
METHODS = BASE_METHODS + CHAIN_METHODS
SCORE_METRICS = tuple(
    f"{method}_{ranking}_{metric}"
    for method in METHODS for ranking in RANKINGS for metric in ("r1", "r5")
)


@dataclass
class Page:
    checkpoint: Path
    records: list[dict[str, Any]]
    all_records: list[dict[str, Any]]
    epoch: int
    subject: int
    config: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.checkpoint.parent.name}/{self.checkpoint.stem}"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", dest="use_all", action="store_true")
    selection.add_argument("--selected-only", dest="use_all", action="store_false")
    parser.set_defaults(use_all=None)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--kind", choices=("best", "last", "both", "score_r1", "all_score"))
    parser.add_argument("--checkpoints", nargs="+", type=Path)
    parser.add_argument("--history-only", action="store_true", help="Open pages without SCORE evaluation.")
    parser.add_argument("--list", action="store_true", help="List pages and exit without opening a window.")
    return parser.parse_args()


def _paths(arguments: argparse.Namespace) -> list[Path]:
    explicit = arguments.checkpoints or list(SELECTED_CHECKPOINTS)
    use_all = arguments.use_all
    if use_all is None:
        use_all = False if arguments.checkpoints else USE_ALL_CHECKPOINTS
    kind = arguments.kind or ALL_CHECKPOINT_KIND
    if use_all and kind in {"score_r1", "all_score"}:
        root = workspace_path(arguments.root or CHECKPOINT_ROOT).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Checkpoint root does not exist: {root}")
        paths = sorted(root.glob("*/best_score_*_*.pt"))
        if kind == "all_score":
            paths += sorted(root.glob("*/best.pt")) + sorted(root.glob("*/last.pt"))
        paths = [path.resolve() for path in paths if path.is_file()]
    else:
        selection = argparse.Namespace(
            root=arguments.root or CHECKPOINT_ROOT,
            kind=kind,
            checkpoints=explicit,
            use_all=use_all,
        )
        paths = _selected_paths(selection)
    if use_all and SORT_PAGES_BY_DATE:
        def date_key(path: Path) -> tuple[str, str]:
            match = re.search(r"(\d{8}_\d{6})$", path.parent.name)
            return (match.group(1) if match else path.parent.name, path.name)
        paths.sort(key=date_key)
    return paths


def _read_history(path: Path) -> list[dict[str, Any]]:
    source = path.parent / "history.json"
    if not source.is_file():
        raise FileNotFoundError(f"History is missing: {source}")
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"History is empty or invalid: {source}")
    return [row for row in records if isinstance(row, dict) and isinstance(row.get("epoch"), int)]


def _checkpoint_epoch(path: Path, history: list[dict[str, Any]]) -> int:
    if path.name == "last.pt":
        return int(history[-1]["epoch"])
    if path.name == "best.pt":
        return int(max(history, key=lambda row: float(row.get("validation_test_r1", -math.inf)))["epoch"])
    if path.name == "best_score.pt":
        keys = [f"validation_score_{method}_{ranking}_r1"
                for method in ("random", "csls_selected") for ranking in RANKINGS]
        return int(max(history, key=lambda row: max(float(row.get(key, -math.inf)) for key in keys))["epoch"])
    match = re.fullmatch(r"best_score_(uncalibrated|random|csls_selected)_(cosine|csls)\.pt", path.name)
    if match:
        key = f"validation_score_{match.group(1)}_{match.group(2)}_r1"
        winner = max(
            (row for row in history if isinstance(row.get(key), (float, int))),
            key=lambda row: float(row[key]), default=None,
        )
        if winner is None:
            raise ValueError(f"No {key} history for {path.name}")
        return int(winner["epoch"])
    raise ValueError(f"Cannot infer epoch for {path.name}")


def load_pages(paths: list[Path]) -> list[Page]:
    pages: list[Page] = []
    seen_epochs: set[tuple[Path, int]] = set()
    for path in paths:
        try:
            config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
            history = _read_history(path)
            epoch = _checkpoint_epoch(path, history)
            identity = (path.parent, epoch)
            if identity in seen_epochs:
                continue
            records = [row for row in history if int(row["epoch"]) <= epoch]
            if not records:
                raise ValueError("checkpoint epoch has no history")
            pages.append(Page(path, records, history, epoch,
                              int(config["data"]["held_out_subject"]), config))
            seen_epochs.add(identity)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Skipping {path}: {exc}", flush=True)
    if not pages:
        raise RuntimeError("No selected checkpoints have usable histories and configs")
    return pages


def _score_settings() -> dict[str, Any]:
    return {
        "enabled": True,
        "pool_class_count": SCORE_POOL_CLASSES,
        "landmark_count": SCORE_LANDMARKS,
        "seed": SCORE_SEED,
        "regularization_ratio": SCORE_RHO,
        "csls_neighbors": SCORE_CSLS_NEIGHBORS,
    }


def _validate_map_stages() -> None:
    for stage in SCORE_MAP_STAGES:
        if len(stage) not in (2, 3):
            raise ValueError("SCORE_MAP_STAGES entries need method, count, and optional rho")
        method, count = stage[:2]
        rho = stage[2] if len(stage) == 3 else SCORE_RHO
        if (method not in {"random", "csls_selected"}
                or not isinstance(count, int) or isinstance(count, bool) or count < 2
                or not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho < 0):
            raise ValueError("SCORE_MAP_STAGES has an invalid method, count, or rho")
    if any(stage[1] > SCORE_POOL_CLASSES for stage in SCORE_MAP_STAGES):
        raise ValueError("A SCORE_MAP_STAGES entry exceeds SCORE_POOL_CLASSES")


def _cache_paths(page: Page) -> tuple[Path, Path]:
    kind = "score_retrofit_chain" if SCORE_MAP_STAGES else "score_retrofit"
    stem = f"{kind}_{page.checkpoint.stem}"
    return page.checkpoint.parent / f"{stem}.json", page.checkpoint.parent / f"{stem}.pt"


def _fingerprint(page: Page) -> dict[str, Any]:
    stat = page.checkpoint.stat()
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "score_settings": _score_settings(),
        "map_stages": SCORE_MAP_STAGES,
    }


def cached_result(page: Page) -> dict[str, Any] | None:
    summary_path, artifact_path = _cache_paths(page)
    if not summary_path.is_file() or not artifact_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("fingerprint") == _fingerprint(page):
            return summary
    except (OSError, ValueError):
        pass
    return None


def _best_logged_scores(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    best: dict[str, dict[str, float | int]] = {}
    for method in BASE_METHODS:
        for ranking in RANKINGS:
            for metric in ("r1", "r5"):
                name = f"{method}_{ranking}_{metric}"
                key = f"validation_score_{name}"
                valid = (
                    row for row in records
                    if isinstance(row.get(key), (int, float))
                    and math.isfinite(float(row[key]))
                )
                winner = max(valid, key=lambda row: float(row[key]), default=None)
                if winner is not None:
                    best[name] = {"epoch": int(winner["epoch"]), "value": float(winner[key])}
    return best


def _available_run_scores(
    pages: list[Page], results: dict[Path, dict[str, Any] | None]
) -> tuple[list[tuple[str, int, dict[str, float]]], int, int]:
    runs: dict[Path, list[Page]] = {}
    for page in pages:
        runs.setdefault(page.checkpoint.parent, []).append(page)
    completed: list[tuple[str, int, dict[str, float]]] = []
    fully_evaluated = 0
    for siblings in runs.values():
        first = siblings[0]
        logged = _best_logged_scores(first.all_records)
        if logged and not SCORE_MAP_STAGES:
            if not all(name in logged for name in SCORE_METRICS):
                continue
            best = {name: float(logged[name]["value"]) for name in SCORE_METRICS}
            settings = first.config.get("score") or {}
            fully_evaluated += 1
        else:
            available = [page for page in siblings if results.get(page.checkpoint) is not None]
            if not available:
                continue
            best = {}
            for page in available:
                metrics = results[page.checkpoint]["metrics"]
                for name in SCORE_METRICS:
                    value = metrics.get(name)
                    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        break
                    best[name] = max(float(value), best.get(name, -math.inf))
                else:
                    continue
                best = {}
                break
            if len(best) != len(SCORE_METRICS):
                continue
            settings = _score_settings()
            if len(available) == len(siblings):
                fully_evaluated += 1
        setting_label = (
            f"{first.config.get('optimization', {}).get('loss_mode', 'samga')} "
            f"P{settings.get('pool_class_count', '?')} "
            f"L{settings.get('landmark_count', '?')} "
            f"ρ{settings.get('regularization_ratio', '?')} "
            f"k{settings.get('csls_neighbors', '?')} "
            f"s{settings.get('seed', '?')}"
        )
        if SCORE_MAP_STAGES:
            setting_label += " chain " + ">".join(
                f"{'R' if stage[0] == 'random' else 'C'}{stage[1]}"
                f"ρ{stage[2] if len(stage) == 3 else SCORE_RHO:g}"
                for stage in SCORE_MAP_STAGES
            )
        completed.append((setting_label, first.subject, best))
    return completed, len(runs), fully_evaluated


def _global_growth_rows(
    pages: list[Page], results: dict[Path, dict[str, Any] | None]
) -> tuple[list[list[str]], int, int, int]:
    completed, total_runs, fully_evaluated = _available_run_scores(pages, results)
    groups: dict[str, list[dict[str, float]]] = {}
    for setting, _subject, best in completed:
        groups.setdefault(setting, []).append(best)
    rows: list[list[str]] = []
    for setting, runs in groups.items():
        for method in METHODS:
            for ranking in RANKINGS:
                values: list[str] = []
                for metric in ("r1", "r5"):
                    name = f"{method}_{ranking}_{metric}"
                    baseline = f"uncalibrated_cosine_{metric}"
                    average = mean(run[name] for run in runs)
                    delta = mean(run[name] - run[baseline] for run in runs)
                    relative = [
                        100 * (run[name] - run[baseline]) / run[baseline]
                        for run in runs if run[baseline] > 0
                    ]
                    values.extend((
                        f"{100 * average:.2f}%",
                        f"{100 * delta:+.2f}",
                        f"{mean(relative):+.1f}%" if relative else "—",
                    ))
                rows.append([
                    setting if method == METHODS[0] and ranking == RANKINGS[0] else "",
                    f"{method.replace('_', ' ')} / {ranking}",
                    str(len(runs)), *values,
                ])
    return rows, len(completed), total_runs, fully_evaluated


def _dataclass_values(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(cls)}
    return {key: value for key, value in values.items() if key in names}


def _workspace_path(value: str) -> Path:
    path = Path(value)
    return workspace_path(path)


def evaluate_page(page: Page) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("Run this script with the PyTorch environment used for training") from exc

    from things_eeg2.model import EEGEncoderConfig, ModelConfig, SAMGAModel, VisualRouterConfig
    from things_eeg2.score_calibration import (
        ScoreCalibrationConfig, evaluate_chained_score_calibration,
        evaluate_score_calibration, select_support_images,
    )
    from things_eeg2.train import DataConfig, SAMGADataset, collect_score_features
    from things_eeg2.visual_lora import LiveViTFeatures, VisualLoRAConfig

    payload = torch.load(page.checkpoint, map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != page.epoch:
        raise ValueError(f"Checkpoint epoch {payload['epoch']} differs from history epoch {page.epoch}")
    saved = payload.get("config") or page.config
    model_values = saved["model"]
    visual_values = dict(model_values["visual"])
    visual_values["layer_ids"] = tuple(visual_values["layer_ids"])
    model_config = ModelConfig(
        eeg=EEGEncoderConfig(**_dataclass_values(EEGEncoderConfig, model_values["eeg"])),
        visual=VisualRouterConfig(**_dataclass_values(VisualRouterConfig, visual_values)),
    )
    data_values = dict(saved["data"])
    for key in ("dataset_root", "visual_feature_dir"):
        data_values[key] = _workspace_path(data_values[key])
    data_config = DataConfig(**_dataclass_values(DataConfig, data_values))
    if data_config.held_out_subject != page.subject:
        raise ValueError("Checkpoint subject does not match the run config")
    lora_values = dict(saved.get("visual_lora") or {})
    if "checkpoint" in lora_values:
        lora_values["checkpoint"] = _workspace_path(lora_values["checkpoint"])
    if lora_values.get("blocks") is not None:
        lora_values["blocks"] = tuple(lora_values["blocks"])
    lora_config = VisualLoRAConfig(**_dataclass_values(VisualLoRAConfig, lora_values))
    score_config = ScoreCalibrationConfig(**_score_settings())
    support_ids = select_support_images(score_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAMGAModel(model_config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    del payload["model_state_dict"]
    adapter_state = payload.get("visual_lora_state_dict")
    if lora_config.enabled and not adapter_state:
        coarse_epochs = int(saved["optimization"]["coarse_epochs"])
        if lora_config.stage == "both" or page.epoch > coarse_epochs:
            raise ValueError("This fine-stage checkpoint has no saved visual LoRA adapter")

    support_dataset = SAMGADataset(
        data_config, model_config, (page.subject,), "training", support_ids
    )
    test_dataset = SAMGADataset(data_config, model_config, (page.subject,), "test")
    support_loader = DataLoader(support_dataset, batch_size=min(SCORE_POOL_CLASSES, 200), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)
    live_visual = None
    try:
        if adapter_state:
            live_visual = LiveViTFeatures(
                lora_config, data_config.dataset_root, model_config.visual.layer_ids, device
            )
            live_visual.load_adapter_state_dict(adapter_state)
            live_visual.visual.eval()
        del payload
        amp_enabled = device.type == "cuda" and bool(saved["optimization"].get("use_amp", True))
        live_test = (live_visual.test_feature_bank(amp_enabled=amp_enabled)
                     if live_visual is not None else None)
        encoded = collect_score_features(
            model, support_dataset, support_loader, test_dataset, test_loader,
            device, support_ids, live_visual, live_test,
            amp_enabled=amp_enabled,
        )
        metrics, artifacts = evaluate_score_calibration(*encoded, support_ids, score_config)
        if SCORE_MAP_STAGES:
            chain_metrics, chain_artifacts = evaluate_chained_score_calibration(
                *encoded, support_ids, score_config, SCORE_MAP_STAGES,
            )
            metrics.update(chain_metrics)
            artifacts["chain"] = chain_artifacts
        baseline = next(
            (row for row in page.records if int(row["epoch"]) == page.epoch), {}
        )
        old_r1 = baseline.get("validation_test_r1")
        difference = (abs(metrics["uncalibrated_cosine_r1"] - old_r1)
                      if isinstance(old_r1, (float, int)) else None)
        warning = (
            f"Offline baseline differs from recorded R@1 by {difference:.3f}; check visual features."
            if difference is not None and difference > 0.0051 else None
        )
        summary_path, artifact_path = _cache_paths(page)
        artifact = {
            "checkpoint": str(page.checkpoint), "epoch": page.epoch,
            "held_out_subject": page.subject, "score_settings": asdict(score_config),
            "map_stages": SCORE_MAP_STAGES,
            "metrics": metrics, "calibration": artifacts,
        }
        temporary_artifact = artifact_path.with_name(artifact_path.name + ".tmp")
        torch.save(artifact, temporary_artifact)
        temporary_artifact.replace(artifact_path)
        result = {
            "fingerprint": _fingerprint(page),
            "checkpoint": str(page.checkpoint),
            "epoch": page.epoch,
            "held_out_subject": page.subject,
            "support_image_ids": list(support_ids),
            "map_stages": SCORE_MAP_STAGES,
            "metrics": metrics,
            "recorded_baseline_r1": old_r1,
            "baseline_warning": warning,
            "calibration_artifact": str(artifact_path),
        }
        temporary_summary = summary_path.with_name(summary_path.name + ".tmp")
        temporary_summary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary_summary.replace(summary_path)
        return result
    finally:
        if live_visual is not None:
            live_visual.close()
            del live_visual
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


class ScoreViewer:
    def __init__(self, pages: list[Page]) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        self.plt = plt
        self.pages = pages
        self.index = 0
        self.results = {page.checkpoint: cached_result(page) for page in pages}
        self.errors: dict[Path, str] = {}
        self.pending: set[Path] = set()
        self.work: queue.Queue[tuple[Page, bool]] = queue.Queue()
        self.finished: queue.Queue[tuple[Path, dict[str, Any] | None, str | None]] = queue.Queue()
        self.worker = threading.Thread(target=self._work_loop, daemon=True)
        self.worker.start()
        self.global_figure: Any | None = None
        self.global_axes: Any | None = None

        self.figure, panel = plt.subplots(2, 2, figsize=(15.5, 9.0))
        self.r1_axes, self.r5_axes = panel[0]
        self.ic_axes, self.score_axes = panel[1]
        self.figure.subplots_adjust(left=0.07, right=0.97, top=0.89, bottom=0.19,
                                    hspace=0.33, wspace=0.24)
        try:
            self.figure.canvas.manager.set_window_title("THINGS-EEG2 SCORE checkpoint pages")
        except AttributeError:
            pass
        self.page_slider = Slider(
            self.figure.add_axes((0.14, 0.10, 0.72, 0.03)),
            "Checkpoint", 1, max(2, len(pages)), valinit=1, valstep=1,
        )
        self.page_slider.on_changed(lambda value: self.select(int(value) - 1))
        if len(pages) == 1:
            self.page_slider.ax.set_visible(False)
        previous = Button(self.figure.add_axes((0.08, 0.025, 0.13, 0.045)), "Previous")
        next_button = Button(self.figure.add_axes((0.24, 0.025, 0.13, 0.045)), "Next")
        global_table = Button(self.figure.add_axes((0.41, 0.025, 0.16, 0.045)), "Global table")
        evaluate_current = Button(self.figure.add_axes((0.61, 0.025, 0.14, 0.045)), "Evaluate page")
        evaluate_all = Button(self.figure.add_axes((0.79, 0.025, 0.14, 0.045)), "Evaluate all")
        self.buttons = (previous, next_button, global_table, evaluate_current, evaluate_all)
        previous.on_clicked(lambda _event: self.move(-1))
        next_button.on_clicked(lambda _event: self.move(1))
        global_table.on_clicked(lambda _event: self._open_global_table())
        evaluate_current.on_clicked(lambda _event: self.enqueue(self.pages[self.index], force=True))
        evaluate_all.on_clicked(lambda _event: self.enqueue_all())
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.timer = self.figure.canvas.new_timer(interval=400)
        self.timer.add_callback(self._drain_finished)
        self.timer.start()
        self.draw()

    def _work_loop(self) -> None:
        while True:
            page, force = self.work.get()
            try:
                result = None if force else cached_result(page)
                if result is None:
                    print(f"Evaluating SCORE: {page.label} (epoch {page.epoch})", flush=True)
                    result = evaluate_page(page)
                    print(f"Saved SCORE: {_cache_paths(page)[0]}", flush=True)
                self.finished.put((page.checkpoint, result, None))
            except Exception as exc:
                self.finished.put((page.checkpoint, None, f"{type(exc).__name__}: {exc}"))
                print(f"SCORE failed for {page.label}: {exc}", flush=True)
            finally:
                self.work.task_done()

    def enqueue(self, page: Page, *, force: bool = False) -> None:
        if page.checkpoint in self.pending:
            return
        if self.results.get(page.checkpoint) is not None and not force:
            return
        self.pending.add(page.checkpoint)
        self.errors.pop(page.checkpoint, None)
        self.work.put((page, force))
        if page is self.pages[self.index]:
            self.draw()

    def enqueue_all(self) -> None:
        for page in self.pages:
            self.enqueue(page)

    def _drain_finished(self) -> None:
        changed = False
        global_changed = False
        while True:
            try:
                path, result, error = self.finished.get_nowait()
            except queue.Empty:
                break
            self.pending.discard(path)
            if error is None:
                self.results[path] = result
            else:
                self.errors[path] = error
            if path.parent == self.pages[self.index].checkpoint.parent:
                changed = True
            global_changed = True
        if changed:
            self.draw()
        if global_changed and self.global_figure is not None:
            self._draw_global_table()

    def _open_global_table(self) -> None:
        if self.global_figure is None or not self.plt.fignum_exists(self.global_figure.number):
            self.global_figure, self.global_axes = self.plt.subplots(figsize=(17, 8))
            self.global_figure.canvas.mpl_connect(
                "close_event", lambda _event: setattr(self, "global_figure", None)
            )
            try:
                self.global_figure.canvas.manager.set_window_title("Global SCORE growth")
            except AttributeError:
                pass
        self._draw_global_table()
        self.plt.show(block=False)

    def _draw_global_table(self) -> None:
        if self.global_figure is None or self.global_axes is None:
            return
        axes = self.global_axes
        axes.clear()
        axes.axis("off")
        rows, available, total, fully_evaluated = _global_growth_rows(self.pages, self.results)
        axes.set_title(
            f"Mean growth versus uncalibrated cosine | {available}/{total} runs with data; "
            f"{fully_evaluated} with all selected scores",
            loc="left", fontsize=15, pad=18,
        )
        if not rows:
            axes.text(
                0.02, 0.75,
                "No runs with SCORE results yet. Evaluate all selected checkpoints, or open a run "
                "with SCORE metrics logged at every epoch.",
                transform=axes.transAxes, va="top", fontsize=11, wrap=True,
            )
        else:
            table = axes.table(
                cellText=rows,
                colLabels=["SCORE setup", "Retrieval mode", "Runs", "Best R@1",
                           "ΔR@1 (pp)", "ΔR@1 (%)", "Best R@5", "ΔR@5 (pp)", "ΔR@5 (%)"],
                cellLoc="left", colLoc="left", loc="upper left",
                colWidths=[0.23, 0.18, 0.05, 0.08, 0.09, 0.09, 0.08, 0.09, 0.11],
                bbox=[0.0, 0.16, 1.0, 0.77],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9 if len(rows) <= 12 else 7)
            for index, row in enumerate(rows, start=1):
                if row[1] == "uncalibrated / cosine":
                    for column in range(9):
                        table[index, column].set_facecolor("#eaf0f8")
        axes.text(
            0.01, 0.12,
            "Each run counts once. Each mode and baseline uses its own best epoch; "
            "Δ is the mean paired difference. Results are provisional until all selected weights are evaluated.",
            transform=axes.transAxes, fontsize=9, color="#444444", wrap=True,
        )
        axes.text(
            0.01, 0.075,
            "Percentage growth averages per-run ratios; baseline R@k = 0 is omitted from "
            "that percentage only. Repeated subjects count as separate runs; setups stay separate.",
            transform=axes.transAxes, fontsize=9, color="#444444", wrap=True,
        )
        axes.text(
            0.01, 0.03,
            "Setup: P = support pool, L = landmarks, ρ = identity regularization, "
            "k = CSLS neighbors, s = random seed.",
            transform=axes.transAxes, fontsize=9, color="#444444", wrap=True,
        )
        self.global_figure.canvas.draw_idle()

    def select(self, index: int) -> None:
        index = max(0, min(index, len(self.pages) - 1))
        if index != self.index:
            self.index = index
            self.draw()

    def move(self, delta: int) -> None:
        self.page_slider.set_val(1 + ((self.index + delta) % len(self.pages)))

    def _on_key(self, event: Any) -> None:
        if event.key in {"right", "down", "pagedown"}:
            self.move(1)
        elif event.key in {"left", "up", "pageup"}:
            self.move(-1)
        elif event.key == "home":
            self.page_slider.set_val(1)
        elif event.key == "end":
            self.page_slider.set_val(len(self.pages))

    def _on_scroll(self, event: Any) -> None:
        self.move(1 if event.button == "down" else -1)

    @staticmethod
    def _series(page: Page, key: str) -> tuple[list[int], list[float]]:
        pairs = [(int(row["epoch"]), float(row[key])) for row in page.records
                 if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
        return [epoch for epoch, _ in pairs], [value for _, value in pairs]

    def _history_panel(self, axes: Any, page: Page, suffix: str, title: str) -> None:
        axes.clear()
        plotted = False
        for key, label, color in (
            (f"validation_test_{suffix}", "Held-out subject / test images", "#1565c0"),
            (f"validation_train_subjects_test_{suffix}", "Training subjects / test images", "#ef6c00"),
            (f"validation_training_{suffix}", "Held-out subject / training images", "#00897b"),
        ):
            epochs, values = self._series(page, key)
            if epochs:
                axes.plot(epochs, values, label=label, color=color, linewidth=2)
                plotted = True
        axes.axvline(page.epoch, color="#5d4037", linewidth=1.1, linestyle="--", alpha=0.7)
        axes.set(title=title, xlabel="Epoch", ylabel=title, ylim=(0.0, 1.0))
        axes.grid(alpha=0.25)
        if plotted:
            axes.legend(loc="best", fontsize=8)

    def _score_panel(self, page: Page) -> None:
        axes = self.score_axes
        axes.clear()
        axes.axis("off")
        logged = _best_logged_scores(page.all_records)
        if logged and not SCORE_MAP_STAGES:
            best = logged
            epoch_count = sum("validation_score_uncalibrated_cosine_r1" in row
                              for row in page.all_records)
            axes.set_title(f"Best SCORE across {epoch_count} logged training epochs", loc="left")
        else:
            siblings = [candidate for candidate in self.pages
                        if candidate.checkpoint.parent == page.checkpoint.parent]
            available = [candidate for candidate in siblings
                         if self.results.get(candidate.checkpoint) is not None]
            best = {}
            for candidate in available:
                scores = self.results[candidate.checkpoint]["metrics"]
                for name, value in scores.items():
                    if (name not in best or float(value) > best[name]["value"]):
                        best[name] = {"epoch": candidate.epoch, "value": float(value)}
            axes.set_title(
                f"Best SCORE among {len(available)}/{len(siblings)} saved checkpoints",
                loc="left",
            )
        if not best:
            pending = any(candidate.checkpoint in self.pending for candidate in self.pages
                          if candidate.checkpoint.parent == page.checkpoint.parent)
            message = ("Queued or evaluating saved checkpoints." if pending else
                       self.errors.get(page.checkpoint, "Select Evaluate all to compute SCORE."))
            axes.text(0.02, 0.75, message, transform=axes.transAxes, va="top", wrap=True)
            return

        def display(name: str) -> str:
            winner = best.get(name)
            return (f"{winner['value']:.3f} (e{winner['epoch']})"
                    if winner is not None else "—")

        rows = [
            [f"{method.replace('_', ' ')} / {ranking}",
             display(f"{method}_{ranking}_r1"),
             display(f"{method}_{ranking}_r5")]
            for method in METHODS for ranking in RANKINGS
        ]
        table = axes.table(
            cellText=rows, colLabels=["Calibration / ranking", "Best R@1 (epoch)", "Best R@5 (epoch)"],
            cellLoc="left", colLoc="left", loc="upper left",
            colWidths=[0.42, 0.27, 0.27], bbox=[0.0, 0.16, 0.98, 0.75],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        best_r1 = max(float(best[name]["value"]) for name in best if name.endswith("_r1"))
        best_r5 = max(float(best[name]["value"]) for name in best if name.endswith("_r5"))
        for index, (method, ranking) in enumerate(
            ((method, ranking) for method in METHODS for ranking in RANKINGS), start=1
        ):
            r1 = best.get(f"{method}_{ranking}_r1")
            r5 = best.get(f"{method}_{ranking}_r5")
            if r1 is not None and float(r1["value"]) == best_r1:
                table[index, 1].set_facecolor("#d9f2df")
            if r5 is not None and float(r5["value"]) == best_r5:
                table[index, 2].set_facecolor("#dce9fa")
        current = self.results.get(page.checkpoint)
        warning = current.get("baseline_warning") if current else None
        note = (
            "Chain stages may reuse support images. Only saved weights can be rescored."
            if SCORE_MAP_STAGES else
            "Each mode and R@k chooses its own epoch."
            if logged else "Only saved weights can be rescored; intermediate epochs are unavailable."
        )
        axes.text(0.01, 0.05, warning or note,
                  transform=axes.transAxes, fontsize=8, color="darkred" if warning else "#555555")

    def draw(self) -> None:
        page = self.pages[self.index]
        self._history_panel(self.r1_axes, page, "r1", "R@1")
        self._history_panel(self.r5_axes, page, "r5", "R@5")
        self._history_panel(self.ic_axes, page, "ic", "Inter-class top-1")
        self._score_panel(page)
        self.figure.suptitle(
            f"{self.index + 1}/{len(self.pages)}  |  held-out subject {page.subject:02d}  |  "
            f"{page.label}  |  epoch {page.epoch}", fontsize=13,
        )
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        self.plt.show()


def main() -> None:
    _validate_map_stages()
    arguments = _arguments()
    pages = load_pages(_paths(arguments))
    print(f"Loaded {len(pages)} checkpoint pages:")
    for page in pages:
        print(f"  subject {page.subject:02d} epoch {page.epoch:03d} | {page.checkpoint}")
    if arguments.list:
        return
    viewer = ScoreViewer(pages)
    if AUTO_EVALUATE_ALL and not arguments.history_only:
        viewer.enqueue_all()
    viewer.show()


if __name__ == "__main__":
    main()
