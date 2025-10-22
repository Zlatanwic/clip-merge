# Repository Guidelines
## Project Structure & Module Organization
- `train.py` is the main entry point and wires `MultiFocusFusionModel` from `modelv2.py`, the CLIP blocks in `clip_module.py`, and dataset loaders.
- Vision models live in `model.py` and `modelv2.py`; shared helpers belong in `tools/` alongside analysis, metrics, and plotting scripts.
- `datasets.py` expects the MFI-WHU layout `data/MFI-WHU/{source_1,source_2,full_clear}`; run `python split_dataset.py` to create `train/` and `val/` splits.
- Checkpoints, intermediate results, and debug visualizations should stay in `checkpoints_mfiwh_full/`, `results/`, and `debug_results/` to keep the root tidy.

## Build, Test, and Development Commands
- Create a virtual environment: `python -m venv .venv && source .venv/bin/activate` (Windows: `Scripts\\activate`).
- Install dependencies: `pip install -r requirements.txt`.
- Prepare data once the dataset is downloaded: `python split_dataset.py` (edit the hard-coded `data_dir` if needed).
- Launch a training run: `python train.py`; adjust the `P` dictionary at the bottom for paths, epochs, or device preferences.
- Monitor training live with `tensorboard --logdir checkpoints_mfiwh_full/tensorboard_logs`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and snake_case names; keep module-level constants uppercase.
- Prefer cohesive modules: models in `model*.py`, dataset logic in `datasets.py`, utilities in `tools/`, and minimal logic inside runnable scripts.
- Add concise docstrings to public functions and annotate tensor shapes when operations are not obvious.

## Testing Guidelines
- Introduce automated checks with `pytest`; mirror the package layout under a new `tests/` directory (e.g., `tests/test_datasets.py`).
- For regression validation, run `python tools/analysis.py` or `python tools/metrics.py` against recent `.npy` dumps and record key scores.
- Before opening a PR, execute a smoke run of `python train.py` with `num_epochs=1` on a small subset to catch breaking changes quickly.

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects following Conventional Commit prefixes (`feat:`, `fix:`, `refactor:`) for easier changelog generation.
- Keep commits scoped to one concern and explain rationale in the body when tweaking losses, schedules, or data contracts.
- PRs should outline motivation, list experiments or qualitative evidence, link related issues, and include reviewers who own the affected area.

## Data & Checkpoint Hygiene
- Store large artifacts in `checkpoints_mfiwh_full/` or `results/`, prune stale runs, and avoid committing them.
- Never commit raw datasets or private assets; extend `.gitignore` when introducing new artifact folders and note GPU/CUDA requirements alongside saved checkpoints.
