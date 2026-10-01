# BrainSAM
<<<<<<< HEAD
=======

BrainSAM is a cross-species, cross-modality brain structure segmentation model built on top of [SAM 2](https://github.com/facebookresearch/sam2) (Meta, Apache-2.0). It adds a modality/organ prior-embedding mechanism ("Embedding Tree"), a Mixture-of-Adapters (MoA) routing module on the deeper blocks of the Hiera backbone, and a boundary-refinement supervision module. The repository also includes training/evaluation scripts, a PyQt5 desktop annotation tool (`app.py`), and a FastAPI-based web annotation tool (`BrainLynx_web/`).

> This repository is derived from Meta's SAM 2 codebase under the Apache License, Version 2.0. See `LICENSE` and `NOTICE` in the repository root.

## Repository layout

```
BrainSAM/
├── app.py              # PyQt5 desktop annotation/inference tool (main, working entry point)
├── BrainLynx_web/      # FastAPI web annotation tool — see its own README
├── Inference/          # Shared inference helpers used by app.py (and, partially, evaluate/)
├── sam2/               # Core model code, adapted from Meta's SAM2 (includes BrainSAM's new modules)
├── training/           # Training entry point, loss functions, optimizer, trainer
├── evaluate/           # Batch inference + metric scripts — see "Evaluation: known issues" below
├── data/               # Dataset wrappers (BrainSAM's own + SAM2's sam2_dataset/)
├── utils/              # NIfTI I/O and misc helpers
├── ui/                 # Qt Designer resources — index.ui is the current main-window layout
├── assets/             # Sample NIfTI volumes (human/monkey/mouse/rabbit) used as demo data
├── work_dir/           # Where checkpoints are expected to live locally (git-ignored)
├── requirements.txt
├── setup.py
├── LICENSE             # Apache-2.0 (required because sam2/ is derived from SAM2)
└── NOTICE              # Third-party attribution + summary of BrainSAM's own modifications
```

## Installation

```bash
conda create -n brainsam python=3.10
conda activate brainsam
pip install -r requirements.txt
# or: pip install -e .   (setup.py declares optional extras: web / gui / train)
```

`sam2/csrc/connected_components.cu` is a CUDA extension source file; `setup.py` does not currently build it, so if you need that op you'll need to add the corresponding build step yourself.

## Desktop app (`app.py`)

The app deliberately does **not** auto-load any model or checkpoint at startup — this lets the window open and the UI be inspected without a GPU or any checkpoint file present. Model loading is manual, via the "Video:" / "PNG:" row at the bottom of the window:

1. Click the "..." button next to **Video:** or **PNG:** to pick a checkpoint file. The dialog opens in `work_dir/` by default.
2. Click **init_v** (video predictor) or **init_p** (image predictor) to actually build that model from the selected checkpoint. Until you click Init, both predictors are `None` and segmentation is unavailable.
3. The **Device** label at the bottom reflects whichever device was actually used the last time a predictor was (re)built — it auto-detects CUDA availability and falls back to CPU rather than being hardcoded.

Two environment quirks already handled in `app.py`, worth knowing about if you touch the top of the file:

- **`OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized`** (Windows): this is a duplicate-OpenMP-runtime conflict between `torch`/`numpy`/`SimpleITK`/`cv2`, not an application bug. `app.py` sets `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` as the very first thing in the file, before any of those libraries are imported — keep it there; moving it below those imports silently brings the crash back.
- **`UnicodeDecodeError: 'gbk' codec can't decode ...`** when reading `.qss` files: on Chinese Windows, `open()` defaults to the system codepage (GBK) rather than UTF-8. The `.qss`/text-file reads in `app.py` pass `encoding='utf-8'` explicitly — do the same for any new text file you read.

## Web annotation tool (`BrainLynx_web/`)

See `BrainLynx_web/README.md` for install/run instructions and known caveats. Note that its `server.py` still has a top-of-file `CONFIG` block (`PROJECT_ROOT`, `SAM2_CKPT`, etc.) that needs to be pointed at your own local paths before running.

## Training

The training entry point is `training/train.py`, which uses Hydra to read YAML configs from `sam2/configs/sam2.1_training/` (default: `sam2.1_hiera_b+BrainSAM.yaml`).

```bash
python training/train.py --config sam2.1_training/sam2.1_hiera_b+BrainSAM.yaml
```

Data paths in that YAML (`img_folder`, `gt_folder`, the per-species `file_list_txt` entries) are placeholders (`/path/to/your/...`) and must be edited for your own data before training. The checkpoint path and the `training/assets/*.txt` split files are already repo-relative and work as-is.

## Evaluation: known issues

`evaluate/config.py` (`EXPERIMENTS` / `get_args()`) holds research-stage experiment configs with placeholder paths — edit it for your own runs. Beyond that, a few scripts in `evaluate/` currently have unresolved imports and will not run as-is:

- `evaluate/evaluate.py` and `evaluate/evaluate_brainSAM.py` import from a lowercase `inference.base_inference`, but the package in this repo is `Inference/` (capital "I"). This happens to resolve on case-insensitive filesystems (Windows, default macOS) but raises `ModuleNotFoundError` on Linux or in CI. Fix by importing `Inference.base_inference` instead (or renaming the package — just be consistent).
- `evaluate/evaluate_brainSAM.py` also imports `build_brainsam_predictor` from `inference.base_inference`, but that function isn't defined anywhere in `Inference/base_inference.py` — it needs to be added, or the import changed to wherever predictor construction actually lives (e.g. `sam2.build_sam.build_sam2_video_predictor`).
- `evaluate/correction_evaluate.py` imports `inference.click_correction`, which doesn't exist anywhere in this repository — this script cannot currently run.

None of the above affects the desktop app (`app.py`) or the training pipeline (`training/`), which are the actively working entry points.

## License

Apache License, Version 2.0 — see `LICENSE`. Most files under `sam2/` retain Meta's original copyright header because they are adapted from [SAM 2](https://github.com/facebookresearch/sam2); `NOTICE` summarizes what was changed for BrainSAM. `NOTICE` still has a `[TODO: your name / lab / organization]` placeholder — fill that in with the actual author/affiliation before publishing.
`sam2/` 目录下大部分文件的版权头仍保留 Meta 的原始声明，因为这些文件是基于 [SAM 2](https://github.com/facebookresearch/sam2)（Apache-2.0）修改而来。完整协议见 `LICENSE`，修改说明见 `NOTICE`——上传前请把 `NOTICE` 里的 `[TODO: your name / lab / organization]` 替换成实际的作者/机构信息。
>>>>>>> 7b6dcc5 (init)
