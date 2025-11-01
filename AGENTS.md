# Repository Guidelines

## Project Structure & Module Organization
The `mmaction/` package carries the training loop, models, datasets, and utilities; treat it as the source of truth when adding core logic. Experiment configs live in `configs/`, grouped by task (`recognition/`, `detection/`, etc.) and mirroring submodules. Command-line helpers sit in `tools/` (single-GPU `train.py`, `test.py`, data conversion), while `tools/analysis_tools/` hosts log and FLOP analyzers. Reference demos and notebooks are under `demo/`, documentation sources under `docs/`, reusable resources in `resources/`, and unit tests in `tests/`. Temporary outputs belong in `work_dirs/` and should stay out of source control.

## Build, Test, and Development Commands
Install dependencies into an activated environment with `pip install -v -e .` and, when needed, extra components via `pip install -r requirements/optional.txt`. Launch training with `python tools/train.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py`. Evaluate checkpoints using `python tools/test.py <config> <checkpoint> --eval top_k_accuracy`. Distributed jobs can be started through `./tools/dist_train.sh <config> <num_gpus>` and the matching `dist_test.sh`. Regenerate analysis artifacts with scripts in `tools/analysis_tools/`, e.g., `python tools/analysis_tools/analyze_logs.py`.

## Coding Style & Naming Conventions
Follow PEP 8: four-space indentation, 79-character lines, and snake_case module names. `setup.cfg` configures `yapf` and `isort`; run them before submitting (`yapf -ir mmaction tests` and `isort mmaction tests`). Classes stay in PascalCase, arguments snake_case, and registries/components should align with existing naming (`Recognizer3D`, `TSNHead`, etc.). Config files follow `<task>/<model>/<variant>.py`; mirror existing directories when adding a new recipe.

## Testing Guidelines
Unit tests rely on `pytest`. Run `python -m pytest tests` before opening a pull request, or narrow scope with `-k` and `tests/path/test_file.py` for quicker feedback. When adding GPU-heavy logic, provide minimal CPU-friendly smoke tests and skip expensive cases conditionally (see `tests/models/backbones/test_resnet3d.py` for patterns). Document expected metrics and attach logs from `work_dirs/` when reporting training results.

## Commit & Pull Request Guidelines
History uses short summaries (e.g., `初始化备份`); keep subjects under 72 characters, prefer English imperative mood, and include scope prefixes when helpful (`train:`, `docs:`). Each pull request should describe motivation, datasets/configs touched, and verification commands. Link related issues, attach evaluation metrics or demo outputs, and share environment details via `python -c "from mmaction.utils import collect_env; print(collect_env())"`. Request review only after CI and local tests pass.
