"""Browse saved training histories without changing checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from things_eeg2.paths import workspace_path
import random
import re
from typing import Any, Sequence


CHECKPOINT_ROOT = workspace_path("checkpoints/things_eeg2_samga_loso_rank192")


USE_ALL_CHECKPOINTS = True
ALL_CHECKPOINT_KIND = "last"


SELECTED_CHECKPOINTS: tuple[Path, ...] = (

)


SAVE_ALL_GRAPHS_TO: Path | None = None


class RunHistory:
    def __init__(
        self,
        checkpoint: Path,
        label: str,
        records: list[dict[str, Any]],
    ) -> None:
        self.checkpoint = checkpoint
        self.label = label
        self.records = records


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare epoch-level metrics from multiple checkpoints."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all",
        dest="use_all",
        action="store_true",
        help="Discover checkpoints from every run under --root.",
    )
    selection.add_argument(
        "--selected-only",
        dest="use_all",
        action="store_false",
        help="Use only --checkpoints (or SELECTED_CHECKPOINTS).",
    )
    parser.set_defaults(use_all=None)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        help="Checkpoint files or run directories to compare.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root containing timestamped checkpoint run directories.",
    )
    parser.add_argument(
        "--kind",
        choices=("last", "best", "both"),
        default=None,
        help="Which checkpoint file(s) to discover with --all.",
    )
    parser.add_argument(
        "--save-all",
        type=Path,
        default=None,
        help="Save one PNG per metric to this directory.",
    )
    return parser.parse_args()


def _resolve_explicit_path(path: Path) -> Path:
    path = workspace_path(path.expanduser()).resolve()
    if path.is_dir():
        path = path / "last.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return path


def _discover_checkpoints(root: Path, kind: str) -> list[Path]:
    root = workspace_path(root.expanduser()).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")
    names = ("last.pt", "best.pt") if kind == "both" else (f"{kind}.pt",)
    discovered = [
        path.resolve()
        for name in names
        for path in root.glob(f"*/{name}")
        if path.is_file()
    ]
    return sorted(discovered, key=lambda path: (path.parent.name, path.name))


def _selected_paths(arguments: argparse.Namespace) -> list[Path]:
    root = arguments.root or CHECKPOINT_ROOT
    kind = arguments.kind or ALL_CHECKPOINT_KIND
    explicit = arguments.checkpoints or list(SELECTED_CHECKPOINTS)
    if arguments.use_all is None:


        use_all = False if arguments.checkpoints else USE_ALL_CHECKPOINTS
    else:
        use_all = arguments.use_all

    if use_all:
        paths = _discover_checkpoints(root, kind)
    else:
        if not explicit:
            raise ValueError(
                "No checkpoints selected. Add paths to SELECTED_CHECKPOINTS, pass "
                "--checkpoints, or enable --all."
            )
        paths = [_resolve_explicit_path(path) for path in explicit]


    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    if not unique:
        raise ValueError("No checkpoint files were found.")
    return unique


def _read_json_history(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"History must be a list: {path}")
    return [record for record in value if isinstance(record, dict)]


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            f"PyTorch is required to read {path.name}. Run this script from the "
            "same Python environment used for training."
        ) from error
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {path}")
    return value


def _history_from_checkpoint(path: Path) -> list[dict[str, Any]]:
    sibling_history = path.parent / "history.json"


    if path.name == "last.pt" and sibling_history.is_file():
        return _read_json_history(sibling_history)

    checkpoint = _load_checkpoint_payload(path)
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    embedded = checkpoint.get("history")
    if isinstance(embedded, list):
        records = [record for record in embedded if isinstance(record, dict)]
    elif sibling_history.is_file():
        records = _read_json_history(sibling_history)
    else:
        metrics = checkpoint.get("metrics")
        records = [metrics] if isinstance(metrics, dict) else []

    return [
        record
        for record in records
        if int(record.get("epoch", 0)) <= checkpoint_epoch
    ]


def _checkpoint_label(path: Path, include_kind: bool) -> str:
    label = path.parent.name
    if include_kind:
        label = f"{label}/{path.stem}"
    return label


def load_histories(paths: Sequence[Path]) -> list[RunHistory]:
    parent_counts: dict[Path, int] = {}
    for path in paths:
        parent_counts[path.parent] = parent_counts.get(path.parent, 0) + 1

    histories: list[RunHistory] = []
    failures: list[str] = []
    for path in paths:
        try:
            records = _history_from_checkpoint(path)
            if not records:
                raise ValueError("no epoch records")
            histories.append(
                RunHistory(
                    checkpoint=path,
                    label=_checkpoint_label(path, parent_counts[path.parent] > 1),
                    records=records,
                )
            )
        except Exception as error:
            failures.append(f"  {path}: {error}")

    if failures:
        print("Skipped checkpoint histories that could not be read:")
        print("\n".join(failures))
    if not histories:
        raise RuntimeError("None of the selected checkpoints contains usable history.")
    return histories


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def available_metrics(histories: Sequence[RunHistory]) -> list[str]:
    metrics = {
        key
        for history in histories
        for record in history.records
        for key, value in record.items()
        if key.endswith(("_r1", "_r5", "_ic")) and _is_finite_number(value)
    }
    preferred = (
        "validation_test_r1",
        "validation_test_r5",
        "validation_test_ic",
        "validation_training_r1",
        "validation_training_r5",
        "validation_training_ic",
        "validation_train_subjects_test_r1",
        "validation_train_subjects_test_r5",
        "validation_train_subjects_test_ic",
    )
    ordered = [metric for metric in preferred if metric in metrics]
    ordered.extend(sorted(metrics.difference(ordered)))
    if not ordered:
        raise ValueError("The selected histories contain no numeric metrics.")
    return ordered


def _metric_display_name(metric: str) -> str:
    return metric.replace("validation_", "validation: ").replace("_", " ")


def _safe_filename(metric: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", metric).strip("_") + ".png"


class MetricViewer:
    def __init__(
        self,
        histories: Sequence[RunHistory],
        metrics: Sequence[str],
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons

        self.plt = plt
        self.histories = list(histories)
        self.metrics = list(metrics)
        self.metric_index = 0
        self.visible_checkpoints = {
            history.checkpoint: True for history in self.histories
        }


        color_map = plt.get_cmap("turbo")
        denominator = max(1, len(self.histories) - 1)
        color_positions = [index / denominator for index in range(len(self.histories))]
        random.Random(20_260_920).shuffle(color_positions)
        self.run_styles = {
            history.checkpoint: {
                "color": color_map(color_positions[index]),
            }
            for index, history in enumerate(self.histories)
        }

        self.figure, self.axes = plt.subplots(figsize=(16.0, 8.5))
        self.figure.subplots_adjust(left=0.07, right=0.69, bottom=0.14, top=0.91)
        try:
            self.figure.canvas.manager.set_window_title(
                "THINGS-EEG2 checkpoint dynamics"
            )
        except AttributeError:
            pass

        previous_axes = self.figure.add_axes((0.34, 0.035, 0.12, 0.055))
        next_axes = self.figure.add_axes((0.54, 0.035, 0.12, 0.055))
        self.previous_button = Button(previous_axes, "Previous")
        self.next_button = Button(next_axes, "Next")
        self.previous_button.on_clicked(lambda _event: self.move(-1))
        self.next_button.on_clicked(lambda _event: self.move(1))


        checklist_height = min(0.73, max(0.18, 0.032 * len(self.histories)))
        checklist_bottom = 0.16 + max(0.0, 0.73 - checklist_height)
        checklist_axes = self.figure.add_axes(
            (0.715, checklist_bottom, 0.275, checklist_height),
            frameon=True,
        )
        checklist_axes.set_title("Visible checkpoints", fontsize=10, loc="left")
        self.checkpoint_checks = CheckButtons(
            checklist_axes,
            [history.label for history in self.histories],
            [True] * len(self.histories),
        )
        for label in self.checkpoint_checks.labels:
            label.set_fontsize(7.2)
        self.checkpoint_checks.on_clicked(self._toggle_checkpoint)

        all_axes = self.figure.add_axes((0.745, 0.055, 0.085, 0.05))
        none_axes = self.figure.add_axes((0.855, 0.055, 0.085, 0.05))
        self.all_button = Button(all_axes, "All")
        self.none_button = Button(none_axes, "None")
        self.all_button.on_clicked(lambda _event: self._set_all_visible(True))
        self.none_button.on_clicked(lambda _event: self._set_all_visible(False))

        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.draw()

    @property
    def metric(self) -> str:
        return self.metrics[self.metric_index]

    def move(self, offset: int) -> None:
        self.metric_index = (self.metric_index + offset) % len(self.metrics)
        self.draw()

    def _on_key(self, event: Any) -> None:
        if event.key in {"right", "down", "pageup"}:
            self.move(1)
        elif event.key in {"left", "up", "pagedown"}:
            self.move(-1)
        elif event.key == "home":
            self.metric_index = 0
            self.draw()
        elif event.key == "end":
            self.metric_index = len(self.metrics) - 1
            self.draw()

    def _on_scroll(self, event: Any) -> None:
        self.move(1 if event.button == "down" else -1)

    def _toggle_checkpoint(self, label: str) -> None:
        for history in self.histories:
            if history.label == label:
                checkpoint = history.checkpoint
                self.visible_checkpoints[checkpoint] = not self.visible_checkpoints[
                    checkpoint
                ]
                break
        self.draw()

    def _set_all_visible(self, visible: bool) -> None:


        current = self.checkpoint_checks.get_status()
        for index, is_visible in enumerate(current):
            if is_visible != visible:
                self.checkpoint_checks.set_active(index)

    def draw(self) -> None:
        metric = self.metric
        self.axes.clear()
        plotted = 0
        for history in self.histories:
            if not self.visible_checkpoints[history.checkpoint]:
                continue
            points = [
                (int(record["epoch"]), float(record[metric]))
                for record in history.records
                if "epoch" in record and _is_finite_number(record.get(metric))
            ]
            if not points:
                continue
            epochs, values = zip(*points)
            style = self.run_styles[history.checkpoint]
            self.axes.plot(
                epochs,
                values,
                linewidth=1.8,
                color=style["color"],
                linestyle="-",
                label=history.label,
            )
            plotted += 1

        display_name = _metric_display_name(metric)
        self.axes.set_title(
            f"{display_name}  ({self.metric_index + 1}/{len(self.metrics)})",
            fontsize=14,
        )
        self.axes.set_xlabel("Epoch")
        self.axes.set_ylabel(display_name)
        self.axes.grid(True, alpha=0.3)
        if metric.endswith(("_r1", "_r5", "_ic")):
            self.axes.set_ylim(0.0, 1.0)
        if plotted:
            legend = self.axes.legend(
                loc="upper left",
                fontsize=7.5,
                frameon=True,
            )
            legend.set_draggable(True)
        else:
            self.axes.text(
                0.5,
                0.5,
                "No selected checkpoint contains this metric",
                ha="center",
                va="center",
                transform=self.axes.transAxes,
            )
        self.figure.canvas.draw_idle()

    def save_all(self, directory: Path) -> None:
        directory = directory.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        original_index = self.metric_index
        for index, metric in enumerate(self.metrics):
            self.metric_index = index
            self.draw()
            self.figure.savefig(
                directory / _safe_filename(metric),
                dpi=180,
                bbox_inches="tight",
            )
        self.metric_index = original_index
        self.draw()
        print(f"Saved {len(self.metrics)} metric graphs to {directory}")

    def show(self) -> None:
        self.plt.show()


def main() -> None:
    arguments = _parse_arguments()
    paths = _selected_paths(arguments)
    histories = load_histories(paths)
    metrics = available_metrics(histories)

    print(f"Loaded {len(histories)} checkpoint histories:")
    for history in histories:
        first_epoch = int(history.records[0].get("epoch", 0))
        last_epoch = int(history.records[-1].get("epoch", 0))
        print(
            f"  {history.label}: epochs {first_epoch}-{last_epoch} "
            f"({history.checkpoint})"
        )
    print(
        f"Available metrics: {len(metrics)}. Use Previous/Next, arrow keys, or "
        "the mouse wheel to switch graphs. Use the checkpoint checkboxes to "
        "show or hide runs."
    )

    viewer = MetricViewer(histories, metrics)
    save_directory = arguments.save_all or SAVE_ALL_GRAPHS_TO
    if save_directory is not None:
        viewer.save_all(save_directory)
    viewer.show()


if __name__ == "__main__":
    main()
