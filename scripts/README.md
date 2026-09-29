# Script guide

Run scripts from the repository root. Each Python command provides `--help`; write generated output to ignored `artifacts/` or another private location.

| Folder | Purpose | Example |
| --- | --- | --- |
| `ops/` | Privacy, release, data integrity and process maintenance | `python scripts/ops/check_repository.py` |
| `spark/` | Optional NVIDIA DGX Spark model download and service setup | `bash scripts/spark/start_model_services.sh` |
| `smoke/` | Short integration checks against configured services or weights | `uv run --no-sync python scripts/smoke/smoke_vision.py` |
| `eval/` | Frozen scoring, comparisons and figures | `uv run --no-sync python scripts/eval/evaluate.py` |
| `data/` | Benchmark data preparation and synthetic fixtures | `uv run --no-sync python scripts/data/make_composite_fixture.py --help` |
| `training/` | Optional reviewer adapter training | `uv run --no-sync python scripts/training/train_grounding_lora.py --help` |

On a CUDA node, `python scripts/eval/benchmark_mask_geometry.py --output artifacts/mask-geometry.json` compares GPU mask reductions with the original CPU scan. It checks exact boxes and overlap decisions before reporting timings. These generated masks measure postprocessing only. For full tracking runs, use `compare_tracking_reports.py --baseline ... --candidate ... --output ... --require-identical-predictions` to require unchanged boxes, states, regions, reasons and confidence values, rather than just equal aggregate accuracy.

`python scripts/eval/benchmark_tracking_capacity.py --output artifacts/tracking-capacity.json` measures 1/2/3/5/10 registered targets at 1/2/5 FPS on a fixed generated 1080p image, with two reversed-order rounds and excluded warmup. Its unpaced queue lag is simulated from measured service times. Add `--paced --objects 1 2 --fps 5 --samples 150 --rounds 1` with a new output path to measure actual paced detector lag. Neither mode includes media decoding, HTTP, database writes or accuracy scoring; use `evaluate_live_replay.py` for the full live path.

Put a new script in the folder matching its role. Keep reusable product behavior in `src/agentx/` and tests in `tests/`; scripts orchestrate those components. Evaluation outputs must distinguish synthetic checks, inspected development data and held out cohorts. Detailed experimental protocols and row level results stay in ignored local storage. Publish only reviewed conclusions with denominators and limits in [competition results](../docs/benchmarks/INDEX.md).
