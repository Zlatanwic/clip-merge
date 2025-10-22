# Repository Guidelines

## Project Structure & Module Organization
`train.py` orchestrates experiments by wiring `MultiFocusFusionModel` from `modelv2.py`, CLIP attention blocks in `clip_module.py`, and dataset utilities in `datasets.py`. Core vision architectures live in `model.py` and `modelv2.py`, while shared helpers, metrics, and plotting scripts reside under `tools/`. Place the raw MFI-WHU assets inside `data/MFI-WHU/{source_1,source_2,full_clear}` and keep generated checkpoints, visualizations, and intermediates in `checkpoints_mfiwh_full/`, `results/`, or `debug_results/` to avoid cluttering the root.

## Build, Test, and Development Commands
Start from a clean virtual environment: `python -m venv .venv && source .venv/bin/activate`. Install dependencies with `pip install -r requirements.txt`. Once the dataset is available, run `python split_dataset.py` to generate `train/` and `val/` splits. Launch training via `python train.py` and adjust the `P` dictionary at the bottom for device selection, epochs, log paths, or subset sizes. Monitor progress in real time with `tensorboard --logdir checkpoints_mfiwh_full/tensorboard_logs`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, snake_case for functions and variables, and UpperCamelCase for classes. Keep module-level constants uppercase and co-locate reusable utilities in `tools/`. Add concise docstrings for public functions, clarifying tensor shapes or modality expectations when not obvious. Stick to ASCII unless a file already uses extended characters.

## Testing Guidelines
Adopt `pytest` with a mirrored layout under `tests/` (e.g., `tests/test_datasets.py`). Name tests after the behaviour under scrutiny (`test_<component>_<condition>`). Before large changes, execute a smoke sanity check by running `python train.py` with `num_epochs=1` and a reduced dataset slice. Record regression metrics using `python tools/metrics.py` against recent `.npy` dumps when validating model updates.

## Commit & Pull Request Guidelines
Compose commits with Conventional prefixes (`feat:`, `fix:`, `refactor:`, etc.) and imperative summaries. Limit each commit to one concern and document rationale in the body when tuning losses, schedules, or data contracts. Pull requests should describe motivation, link related issues, share quantitative or qualitative evidence (attach TensorBoard screenshots or metric tables), and request reviews from owners of the affected modules.

## Data & Artifact Hygiene
Never commit raw datasets, checkpoints, or large binary artifacts. Store heavy outputs in `checkpoints_mfiwh_full/` or `results/`, prune obsolete runs regularly, and update `.gitignore` whenever adding a new artifact directory. Note GPU or CUDA requirements alongside any saved checkpoints to keep reproduction straightforward.
