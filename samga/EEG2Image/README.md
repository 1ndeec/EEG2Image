# SAMGA with visual LoRA for EEG-image retrieval

This project trains a shared EEG-image retrieval model on THINGS-EEG2. It
combines a TSConv EEG encoder with subject-aware routing of intermediate
visual features, and optionally adapts OpenCLIP ViT-B/16 with LoRA.

The main notebook trains nine participants and evaluates the held-out
participant. A second notebook defines visual-adaptation and model ablations.

## Setup

Use Python 3.10 or newer. Install a PyTorch build appropriate for your GPU in
your training environment, then install this project from this directory:

```sh
python -m pip install -e ".[notebooks,plots]"
```

Notebook cells do not install dependencies or reload imported modules. Restart
the kernel after editing Python code during a notebook session.

## Data and pretrained weights

Large files stay outside this code directory:

```text
EEG2Image/
  samga/
    things_eeg2/
    train_things_eeg2.ipynb
  training_data/nm000232/
    derivatives/nice_250hz/sub-XX/
      averaged_eeg_training.npy
      averaged_eeg_test.npy
      metadata.json
    stimuli/
      image_metadata.npy
      training_images.zip
      test_images.zip
      vit_b16_laion2b_layers_2_10_cls/
      vit_b16_laion2b_layers_2_10_patch_mean/
  models/ViT-B-16-laion2b/open_clip_model.safetensors
  checkpoints/
```

Relative paths resolve against the parent of `samga/`, independently of the
working directory. Set `EEG2IMAGE_ROOT` before starting Python to use another
workspace, or provide absolute configuration paths. Previously saved
workspace-relative checkpoint configurations remain readable.

EEG input is `[images, 63, 250]`, with repetitions averaged before training.
The dataset opens arrays read-only. See the
[preprocessing contract](things_eeg2/PREPROCESSING_COMPATIBILITY.md) for row
order and filtering details.

Create or resume both visual feature caches with:

```sh
python -m things_eeg2.extract_vit_b16_features
```

Edit `ExtractionConfig` in that module if needed. Each cached image has a
768-dimensional class-token or mean-patch vector for each ViT block. Keep
extraction manifests beside the arrays so the loader can check completion.

## Training

Open `train_things_eeg2.ipynb` in the installed environment. Its cells configure
the experiment, inspect caches, show parameter counts, and train the requested
folds. Notebook settings override the configuration classes' original
InternViT defaults.

- **Visual adaptation:** enable LoRA during coarse epochs, fine epochs, both,
  or neither. Block numbers are one-based. Q/V adapters are standard; switches
  add K, attention-output/MLP adapters, or trainable LayerNorms.
- **Batching:** select images, then include every training participant's EEG
  for each image. Encode each unique image once. Gradient caching replays
  visual chunks while retaining the full retrieval/MMD batch.
- **Stages:** coarse training mixes retrieval and multi-kernel MMD. Fine
  training freezes the shared alignment encoder and drops MMD.
- **Loss:** `samga` uses diagonal-pair cross-entropy averaged across both
  retrieval directions, with the full similarity matrix. `score` uses
  multi-positive alignment and optional supervised recovery. These are
  separate training configurations.

With recovery enabled, each epoch re-encodes the labeled support pool, selects
pairs separately for every training participant, and fits fresh orthogonal
maps. A pseudo-target participant is excluded from ordinary alignment in each
batch; only its recordings outside the support pool receive recovery loss.
Gradients reach query encoders, but do not pass through map fitting.
`fixed_v1` retains fixed pair selection as an explicit ablation.

Target-subject calibration uses known pairs from training concepts. Test EEG
does not determine moments, pair selection, maps, or the fixed CSLS reference
bank. This supervised protocol differs from original SCORE's unlabeled
target-batch correspondence discovery.

`train_things_eeg2_lora_sweep.ipynb` defines 24 participant-3 ablations. Set its
reference configuration path before use. Fine-only LoRA variants reuse a
separately trained coarse checkpoint; variants changing coarse training or the
main model start from epoch one.

## Checkpoints and inspection

Runs save configuration, epoch history, router weights, and `last.pt`.
`best.pt` tracks uncalibrated Top-1. Calibration also saves per-variant Top-1
winners and the best calibrated checkpoint. LOSO summaries distinguish final
metrics from independently selected epoch peaks.

Automatic resume requires matching settings. Explicit resume can extend a
schedule. Fine-stage LoRA checkpoints must include adapter weights; loading
does not substitute another run's checkpoint or silently initialize missing
adapters.

```sh
python plot_checkpoint_dynamics.py --all
python plot_score_checkpoint_dynamics.py --history-only
```

The SCORE viewer can evaluate saved weights when requested and cache scores
beside checkpoints. It does not update encoder weights.

## Code guide

| Module | Responsibility |
| --- | --- |
| `model.py` | TSConv, visual projections, routing, shared alignment |
| `config.py` | Settings, path resolution, startup validation |
| `data.py` | Read-only arrays, feature caches, batches, smoothing |
| `losses.py` | Full-batch paired retrieval and MMD |
| `visual_lora.py` | ViT adapters, pooled features, gradient-cache replay |
| `score_training.py` | Multi-positive and support-to-query objectives |
| `score_calibration.py` | Moment matching, pair selection, Procrustes, CSLS |
| `train.py` | Stage transitions, evaluation, optimization, checkpoints |
| `experiments.py` | LOSO scheduling, resume discovery, summaries |
| `extract_vit_b16_features.py` | Resumable visual feature extraction |

The import name remains `things_eeg2`. Configuration classes, datasets, and
losses are re-exported from `things_eeg2.train` for existing analysis callers.

## Tests

From this directory:

```sh
python -m unittest discover -s tests -v
```

Tests use synthetic tensors and need no downloaded model or EEG dataset.

The model follows [SAMGA](https://arxiv.org/abs/2604.17782); coordinate recovery
is inspired by [SCORE](https://arxiv.org/abs/2608.19134). Visual LoRA and the
labeled-support protocol are configurable extensions in this project.
