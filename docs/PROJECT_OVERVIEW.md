# AgentX project overview

This is the short technical handoff for contributors. The linked documents cover implementation contracts and the reviewed competition conclusions.

## Purpose and user workflow

AgentX FindBack is a visual memory application for recorded footage. It helps a user find a registered object's last supported location, inspect its history, and replay the original evidence. Answers describe the recording at a chosen time, not an object's present location in the physical world.

The main workflow is: upload a fixed-camera video (or start a live camera capture), register an object with a reference box, build a memory version, ask a location or history question at a cutoff, and inspect the cited frame. A saved answer can be exported as a standalone HTML report with evidence images, provenance and an integrity verifier.

## Main capabilities and boundaries

| Area | Implemented behavior | Boundary |
| --- | --- | --- |
| Video memory | Registration, timestamped observations, movement/visibility events and immutable index runs | Registration does not create observations before its timestamp |
| Questions | Location, last-seen and history answers; bounded agent tools and typed object selection | Published location claims come from verified memory; ambiguous selection remains explicit |
| Live capture | Browser or pushed camera frames appended to a growing recording; memory follows it and answers while it records; stopping seals and hashes it | No final hash while recording, so review and export wait for the seal; one camera per capture |
| Registration | Manual box drawing, or advisory RT-DETR/Cosmos candidates a user confirms | A suggestion is not evidence; discovery writes no observation or registration |
| Evidence | Original-frame replay, source checks, cutoff validation and offline reports | Integrity checks do not prove perception accuracy |
| Perception | SAM 2.1 temporal tracking (the Spark profile, with DINOv2 appearance checks and an RT-DETR look-alike guard); OpenCV reference tracking (weight-free baseline); RT-DETR with ByteTrack; Cosmos grounding | Occlusion and look-alike identity swaps remain difficult |
| Camera pose | Per-frame affine estimation; positions re-read in registration coordinates | A view that cannot be related to the registration frame keeps its box and loses its region |
| Visual review | Saved Cosmos interpretations, including per-frame grounded review | Reviews remain separate from authoritative observations |
| Operations | Persistent jobs, cancellation, schema migrations and optional API-token authentication | One process and one trusted team workspace; no tenant isolation or distributed worker ownership |

## Architecture and code locations

The backend is a Python modular monolith. FastAPI exposes application services; SQLAlchemy and Alembic manage SQLite metadata; original videos and derived media live in the configured data directory. Svelte and TypeScript provide the browser client. Optional model adapters load local weights or call configured inference services.

| Responsibility | Main location |
| --- | --- |
| Time, state and evidence contracts | `src/agentx/domain/` |
| Indexing, questions, selection and evidence reports | `src/agentx/services/` |
| Model calls, bounded tools and query workflow | `src/agentx/agents/` |
| Versioned application instructions | `src/agentx/skills/` |
| Video decoding, scene checks and perception adapters | `src/agentx/vision/` |
| HTTP API, persistence and migrations | `src/agentx/api/`, `storage/`, `migrations/` |
| Browser application and workflows | `web/src/`, `web/tests/` |
| Backend regressions and small fixtures | `tests/` |
| Evaluation, packaging and GPU setup tools | `scripts/`, `scripts/spark/` |

The complete module map is in [docs/README.md](README.md). Read the [architecture and memory contract](ARCHITECTURE.md) before changing time handling, object identity, evidence, model fallbacks or persistence.

## Run and verify

Prerequisites: Python 3.12, uv, Node.js 22.12 or newer, and FFmpeg 6 or newer. From the repository root:

```bash
make setup
cp .env.example .env
make web
uv run --no-sync agentx doctor
uv run --no-sync agentx serve
```

Open `http://127.0.0.1:9000` and choose the controlled sample. The default OpenCV path does not require a hosted model or downloaded model weights. The generated sample validates software behavior, not real-world accuracy. Optional GPU, planner and Cosmos configuration is in [deployment](DEPLOYMENT.md).

For a change, use `make privacy-check`, `make check`, `make test`, `make evaluate`, and the relevant browser/model checks in [CONTRIBUTING.md](../CONTRIBUTING.md). The CI workflow is in `.github/workflows/ci.yml`. Environment credentials stay blank in the public template and are configured locally.

## Evidence and known limits

The [competition results](benchmarks/INDEX.md) summarize real desk clips, live replay, camera recovery, agent Skill comparisons and reviewer alternatives. They state denominators and rejected changes. The [evaluation method](EVALUATION.md) separates generated software checks, public video stress tests and a future original-footage user study. The [validation summary](VALIDATION.md) records which product paths passed controlled checks.

The strongest current evidence is a small frozen real-clip question set (11/11 held-out answers with no wrong location on the final build), paired camera movement samples, scoped generated agent tests and final-build engineering measurements. Those measurements are: a CUDA mask stage with identical predictions, warm live ingestion at 5 FPS for one or two targets, and a target-count capacity matrix. Similar objects, occlusion and false visual-review presence remain open limitations. No broad natural-user accuracy or matched NVIDIA VSS superiority claim is supported.

## Collaboration and local material

Use [CONTRIBUTING.md](../CONTRIBUTING.md) for changes and [SECURITY.md](../SECURITY.md) for vulnerability reports. AgentX source is Apache-2.0; third-party code, fonts, data and model weights retain their own terms, documented in [third-party notices](../THIRD_PARTY_NOTICES.md).

Shared source includes code, tests, dependency locks, public technical documentation, reviewed figures and result summaries. Personal plans and journals stay in ignored `.private/`; credentials stay in `.env`; application data, raw recordings, model weights and generated artifacts stay in their ignored local directories. See [repository privacy](REPOSITORY_PRIVACY.md) before preparing a source upload, especially when existing Git history is involved.
