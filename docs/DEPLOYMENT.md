# Deployment

## Local development

Follow the [quickstart](../README.md#quick-start-no-gpu-no-model-weights). The default OpenCV backend and the generated sample work without model weights or API keys. `agentx doctor` checks the prerequisites. Build the browser client before starting the application. A single process owns the SQLite workspace and the indexing worker.

To let another machine reach the app, set a strong `AGENTX_API_TOKEN` in an ignored `.env` file and serve behind an HTTPS reverse proxy. Browser login uses a same-site HttpOnly cookie; API clients send `Authorization: Bearer ...`. The optional `AGENTX_AGENT_TOKEN` grants read-and-ask access only, for Skills, MCP and OpenClaw agents. Never put tokens in URLs or logs.

Without a token, use a loopback connection and a `localhost` or loopback Host header. Browser writes must come from the application's own origin, including in local development. API clients may authenticate explicitly with a Bearer token. An invalid Authorization header is rejected even if a valid session cookie is present.

## NVIDIA DGX Spark

The competition profile runs on one DGX Spark:

- NVIDIA Nemotron 3.5 Lightning plans the tool steps.
- Cosmos-Reason2-8B reviews frames, optionally with the team's LoRA adapter.
- SAM 2.1 Small tracks objects, with DINOv2 and RT-DETR supporting it for perception.

The approved node already provides an NVIDIA runtime and vLLM. Inspect its installed packages and weights before adding dependencies. The source bundle contains no credentials, media, models or runtime data.

1. Build and verify the reviewed bundle with `make release`. Transfer only the bundle and its checksum to the assigned node.
2. Run `scripts/spark/setup_node.sh` in the project account. It creates a user-level environment from the reachable package mirror; it makes no system or driver changes. The organizer image supplies the Nemotron checkpoint. Use `scripts/spark/download_models.sh` (Cosmos from ModelScope) and `download_sam2.py` (hash-verified SAM 2.1 Small) only for missing, approved files. Small Hugging Face weights (RT-DETR, DINOv2) are copied into the offline cache and used with `HF_HUB_OFFLINE=1`.
3. Run `bash scripts/spark/start_model_services.sh` (or `make node-services`). It starts Nemotron and then Cosmos in project-owned tmux sessions. Before starting the next service, it waits until the current one lists its model name on its loopback `/v1/models`. This matters because vLLM profiles free memory while it starts, and a second engine loading at the same time makes that profiling fail. To serve the adapter, export `COSMOS_LORA_PATH` (the local adapter directory) and `COSMOS_LORA_NAME` (the served name that `AGENTX_COSMOS_REFERENCE_ADAPTER_MODEL` will use) before starting; the script also waits for that name. If a session exits or a model does not appear, read its log under `~/agentx-logs/`.
4. Create `.env` from the blank [template](../.env.example) and apply the [Spark competition profile](#spark-competition-profile) below. Use private absolute paths for `AGENTX_DATA_DIR` and the model paths.
5. Run `scripts/ops/preflight.py --require-cuda`, the relevant `scripts/smoke/` checks, and `agentx doctor --probe`. The doctor sends both planner request shapes (with and without the selection thinking budget) and checks that each configured Cosmos name, including the adapter, is advertised. That listing shows the service is reachable; it does not test visual generation. Build the web client, then start the single application process with `scripts/ops/node_service.sh start` (or `make node-start`).
6. Reach the application through an SSH tunnel to its loopback port, or through an organizer-approved authenticated HTTPS route. Keep the model ports private.

## Spark competition profile

These are the settings behind the published results and the demo. The generic defaults in `.env.example` are more permissive, so they do **not** reproduce this profile.

### Model services

| Service | Checkpoint | Served name and port | Max model length | Max sequences | Other flags | `--gpu-memory-utilization` | Weights |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Planner (`serve_nemotron.sh`) | `Nemotron-3.5-Lightning-30B-A3B-NVFP4`, supplied by the organizer (ModelOpt mixed NVFP4) | `nemotron` on 127.0.0.1:8000 | 32,768 | 8 | `--reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --trust-remote-code` | 0.3 | 17.86 GiB |
| Reviewer (`serve_cosmos.sh`) | `Cosmos-Reason2-8B` from ModelScope `nv-community`, optional LoRA | `nvidia/Cosmos-Reason2-8B` (and `findback-grounding-v1`) on 127.0.0.1:8002 | 16,384 | 4 | `--dtype bfloat16 --limit-mm-per-prompt {"image":8}`; with an adapter, `--enable-lora --max-lora-rank 16 --lora-modules <name>=<path>` | 0.3 | 16.74 GiB |

Both services ran on vLLM 0.28.0 in the organizer image. The memory fractions are shares of the GB10's unified memory, and they add up. With both engines resident, `nvidia-smi` reported 37,002 MiB for the planner engine, 34,623 MiB for the reviewer engine, and 1,374 MiB for the application (SAM 2.1 and DINOv2). Leave headroom for the application and the operating system, which share the same pool. An earlier session ran the planner at 0.22 and the reviewer at 0.42. The current scripts use 0.3 for both, which you can override with `NEMOTRON_GPU_FRACTION` and `COSMOS_GPU_FRACTION`. The FindBack planner sends no OpenAI tools, so the tool-call parser does not change its JSON contract. The reasoning parser is required: without it, vLLM rejects the selection thinking budget, and every indirect description falls back to the deterministic resolver.

### Application settings

Every variable below is defined in `src/agentx/config.py` (prefix `AGENTX_`).

```bash
# Planner: Nemotron via vLLM on loopback
AGENTX_PLANNER_BASE_URL=http://127.0.0.1:8000/v1
AGENTX_PLANNER_MODEL=nemotron
AGENTX_PLANNER_MAX_STEPS=8
AGENTX_PLANNER_MAX_TOKENS=800
AGENTX_PLANNER_THINKING=false
AGENTX_PLANNER_STRUCTURED_OUTPUTS=false
AGENTX_PLANNER_SELECTION_THINKING_BUDGET=512

# Reviewer: Cosmos-Reason2-8B via vLLM on loopback
AGENTX_COSMOS_BACKEND=http
AGENTX_COSMOS_BASE_URL=http://127.0.0.1:8002/v1
AGENTX_COSMOS_MODEL=nvidia/Cosmos-Reason2-8B
AGENTX_COSMOS_REFERENCE_GROUNDING=integer_1000
AGENTX_COSMOS_REFERENCE_MAX_EDGE=768
AGENTX_COSMOS_REVIEW_WORKERS=8
# Optional opt-in reference review: base presence, then the adapter's point
# AGENTX_COSMOS_REFERENCE_ADAPTER_MODEL=findback-grounding-v1

# Perception: guarded SAM 2.1 Small
AGENTX_SAM2_MODEL_PATH=/absolute/path/to/sam2.1-hiera-small
AGENTX_SAM2_PRESENCE_THRESHOLD=0.95
AGENTX_SAM2_CONFIDENT_PRESENCE=0.995
AGENTX_SAM2_APPEARANCE_THRESHOLD=0.85
AGENTX_IDENTITY_ENABLED=true
AGENTX_SAM2_DISTRACTOR_GUARD=true
AGENTX_SAM2_MAX_DISTRACTORS=4
```

Some parts of the profile are not environment variables:

- **Per-run settings.** The tracker and sampling rate are chosen per memory build: in the browser, pick **SAM 2.1 temporal tracking** under Perception. The UI requests 5 samples per second for every backend except Cosmos. Through the API, send `{"backend": "sam2", "sample_fps": 5}` to `POST /api/v1/videos/{id}/runs`.
- **Code defaults.** SAM 2.1 runs in bf16 on CUDA and keeps its session state on the GPU (`AGENTX_SAM2_STATE_DEVICE=auto`). It uses 32 memory frames. The distractor guard's 0.6 collision coverage and 4× mask-area jump are fixed in code. Every completed run records these values in its provenance.

The live camera profile uses the defaults `AGENTX_LIVE_MAX_FPS=10` and `AGENTX_LIVE_MAX_EDGE=1280`; the browser client sends 5 frames per second. The published live measurements used one or two registered objects at 5 FPS. The detector alone manages about 5.6 samples per second with five objects and 3.1 with ten, so larger inventories need a lower sampling rate.

### Reviewer adapter

`AGENTX_COSMOS_REFERENCE_ADAPTER_MODEL` enables an opt-in mode for reference-crop reviews. The base model decides presence first, and the adapter's point is used only when both models say the object is visible. Free-form reviews, registration suggestions and Cosmos indexing keep using the base model. The adapter was trained on the GB10 with `scripts/training/train_grounding_lora.py`. On a prospective bottle cohort it raised correct points from 167 to 185 of 240, but it still made many identity-proxy false claims. Reviews stay advisory. See [results and limits](benchmarks/INDEX.md#visual-reviewer-and-model-tuning).

## Preserve data during upgrades

Keep `.env` and the complete application data directory outside release directories. Before replacing a running version, save a consistent database backup together with its media, then verify it with `scripts/ops/verify_data_snapshot.py`. Old evidence frames need the original source and the saved review bytes; when migrating legacy reviews, run `scripts/ops/backfill_review_frames.py` on the original runtime. Restore into a separate location first, and verify source hashes, references and citations before switching service. Rebuilds create new immutable memory versions.

These scripts should manage only project-owned tmux sessions and user-level files. The organizer's OS, SSH service, users, network and NVIDIA drivers stay untouched.
