# Architecture and memory contract

## Design boundary

FindBack is a modular monolith with two execution paths: video indexing and question answering. `domain/` owns temporal invariants; `services/` owns application operations; HTTP and model libraries are adapters. New applications should reuse the services or `/api/v1`, not import UI state or decode query prose.

The detector protocol returns `Detection(object_id, box, score, reason, track_id)`. A visible box must come from the current decoded frame. Normalized coordinates are relative to the upright source image. Quarter-turn display metadata is applied consistently to decoding, registration, evidence and browser playback. A track ID belongs to a short-lived tracker; a registered object ID belongs to a video inventory. Neither is a global physical-world identity.

## Storage

| Record | Meaning |
| --- | --- |
| Video | Original file hash, metadata, source/proxy paths, current region configuration |
| RegisteredObject | Name, category, reference crop, first registration time, source-frame bounds |
| IndexRun | Input scope, frozen inventory and region names, backend/configuration, status and provenance |
| Observation | One sampled object's visible box or explicit missing/ambiguous result at a source time, with detector score and optional identity similarity |
| Event | An appearance, confirmed regional movement, disappearance, ambiguity or reappearance, linked to its observation |
| QueryRecord | Question, cutoff, final answer, evidence references, warnings, planner model and Skill/tool trace |
| ReviewRecord | A non-authoritative Cosmos interpretation with its frames, hashes, mode and provenance |

The database stores metadata; original video and crops live under `AGENTX_DATA_DIR/videos/<video_id>/`. Observation frames are decoded from the original on demand. Reviews preserve the exact JPEGs supplied to the model under the recording directory, keyed by source hash, timestamp, encoding edge and frame hash. Replay verifies the original source and cached bytes; legacy reviews can backfill only when their original encoded hash reproduces. This keeps review evidence stable across decoder migrations. Playback uses a silent H.264 proxy normalized to start at zero. A JSON export contains a run, observations, events and query responses; it does not embed media.

`(run_id, object_id, at_ms)` is unique for observations and indexed for historical reads. Each rebuild creates a fresh run. Finished runs are not overwritten by changed thresholds or renamed regions. Object edits are deliberately limited to adding a registration; old runs retain their inventory IDs.

The browser's memory-version selector reopens each run with its own saved queries, reviews, region names and object inventory. Switching versions preserves the query cutoff, clears the prior object selection and discards stale asynchronous responses. Later registrations remain outside an older run's query scope.

An answer separates its state at the requested cutoff from any earlier confirmed sighting. An uncertain or last-seen state labels the previous frame as history and offers a jump to the cutoff frame for inspection; that frame is context, not a new positive observation. Evidence boxes are overlaid on the rendered source image at its own aspect ratio, including portrait video, so their normalized coordinates align with the cited pixels.

## Temporal rules

1. Source presentation timestamps, not nominal frame counts, determine time. Input sample times increase strictly.
2. Registration at time T does not provide observations before T.
3. A question at cutoff T can retrieve only records with `at_ms <= T`, inside its chosen run and video.
4. `visible` requires a recent positive observation without a later missing/ambiguous record. Freshness is bounded by the sample cadence.
5. `last_seen` retains a previous positive observation while `current_zone` is null.
6. `unknown` covers detected scene changes, ambiguous or unconfirmed identity, or missing original evidence. An `identity_unconfirmed` candidate can retain an earlier cited sighting but cannot turn that sighting into a location at the cutoff. Old evidence can explain the past, never establish the present.
7. `not_observed` means no positive evidence is available before the cutoff. No location is invented.
8. Regional movement events require two consecutive observations in the new region. Raw observations preserve the actual per-frame region, so event confirmation does not rewrite the past.
9. Overlapping region membership is unassigned. No arbitrary region wins.
10. Original media must exist before an answer calls a location supported. A model interpretation cannot modify observations.

The system reports what was visible **in the recording**, not where an item currently is in the real world. It does not distinguish hiding, removal and detection failure unless independently observed evidence supports that distinction.

## Agent and Skills

`QueryWorkflow.answer` has two planners behind one contract. The **local resolver** is a small deterministic function: registered names/categories and three intents. The **tool agent** (`agents/agent.py`) runs when an OpenAI-compatible planner is configured; on the DGX Spark node that is NVIDIA Nemotron 3.5 Lightning served by vLLM on loopback. Any JSON-mode chat endpoint works, including a hosted StepFun model through the legacy variables.

For simple inventory lookups and explicit object selections, the agent is a bounded loop, not a framework: the model answers one JSON step per turn, either `call_tool` with a tool name and arguments or `final` with an answer, an intent, the resolved object and the observation IDs it cites. Tools execute in code over the same `Memory` service the API uses: `find_objects`, `list_objects`, `get_state`, `get_history`, `get_observation` and a single-use `review_frames` that delegates to Cosmos. Results return to the model as data. Unknown tools, repeated identical calls, more than one review, more than `AGENTX_PLANNER_MAX_STEPS` turns or repeatedly invalid JSON end the loop, and the local resolver answers with a visible warning. Unknown keys are ignored and long text is clipped, so a model cannot extend its own action space.

Indirect and qualified named descriptions use a separate typed translation step. A dedicated `compile-object-selection` application Skill preserves selection criteria; code evaluates all eligible objects and handles zero/one/many matches. A named qualifier cannot be discarded by falling back to a plain name lookup. A positive event-count upper bound requires count language in the question; a singular object does not imply the event happened exactly once. Explicit region-history phrases such as "never left the center" require the corresponding whole-history predicate, rather than one sighting or a current state; "never outside" and "never inside" have different meanings. Invalid plans receive bounded repair feedback rather than silently changing the requested criteria. Opening-frame requests preserve a time-zero state constraint; repair feedback distinguishes that snapshot from an unconstrained first sighting. Registration at the recording start retains a registration predicate with both interval bounds at zero. A singular "only item in the center" request must not invent a whole-history condition, and an explicit "after" interval requires a lower time bound on its history predicate. The generic modifier "registered" cannot become a name filter unless a named object supports that interpretation. These are narrow checks, not full semantic verification. The bounded schema includes another registered object's state, excluding the target itself. Translation remains fallible and requires model evaluation; see [the selection protocol and results](benchmarks/INDEX.md).

Code decides both the evidence and the published location claims. The planner chooses tools, an object and an intent; its proposed prose cannot override memory. Explicit user selection and exact inventory matches take precedence over the planner. Only the selected object appears in the final state and citations. Proposed citations must have been returned to the model, belong to that object, be visible, and precede the cutoff; violations produce a warning. A shared answer renderer then reads the selected state and supported history, validates original media and object/run/time scope, and builds the answer and replay references from those records. Non-visible states retain the last observation and the "not confirmed" sentence. Cosmos interpretations remain separately saved reviews. The trace records the planner model, tool calls, Skill hashes and `answer_source=verified_memory_v1`. These guards apply equally without query Skills.

`GET /api/v1/skills` publishes each Skill's declared version, purpose, SHA-256 and exact body, so the hash recorded on a saved answer can be verified against the running deployment's instructions.

Six Skills are packaged as versioned Markdown:

- `observe-object-events`: observation/event policy; its hash is recorded with every index run. The reducer implements this policy in code; no LLM interprets this Skill during indexing.
- `retrieve-object-history` and `verify-location-answer`: query scope, evidence, ambiguity and uncertainty instructions, injected into both planners' traces.
- `compile-object-selection`: preserve a complete object description in a bounded typed memory query; code determines matches.
- `answer-with-memory-tools`: the tool policy of the agent (state first, history for change questions, one review at most, cite retrieved IDs only).
- `review-visual-evidence`: the review instructions injected into the Cosmos prompt.

`Question.use_skills=false` removes the query and agent Skills from the prompt and the trace; it never disables source/time guards or changes the observation reducer. `Question.use_provider=false` forces the local resolver, which gives a same-cutoff baseline for any agent answer. Those two flags are the arms of the controlled comparison in [the Skills ablation](benchmarks/INDEX.md); [the Skill package guide](../src/agentx/skills/README.md) documents each Skill's triggers, negative triggers and contract.

## Registration discovery

`services/discovery.py` proposes registration candidates for one frame so a person can name an
object instead of drawing its box. This is the third role one Cosmos-Reason2 deployment plays,
beside visual review and the `cosmos` perception backend: the same endpoint and weights answer
"what is in this frame and what would you call it", "where is this described object in this
frame" and "where is every instance of this category", with no additional configuration. `POST /api/v1/videos/{id}/suggestions` reads one original
frame and asks a perception backend what is visible: `rtdetr` returns COCO-category boxes with
scores from a process-cached proposer, and `cosmos` asks Cosmos-Reason2 for an open-vocabulary
listing of named objects, parsed with the same tolerant box reader as the Cosmos adapter.

Discovery is advisory and writes nothing. It creates no observation, event, reference crop or
registered object; the user still confirms each candidate through the unchanged registration
path, so every memory invariant holds regardless of what a model proposed. Proposals are
deduplicated by 0.7 IoU, dropped when the box is too small to be a usable reference, capped at
twelve, and given names that cannot collide with an already-registered object. An
open-vocabulary name is not a detector category, so a Cosmos proposal registers as `custom`
reference tracking unless the user selects a category. The Cosmos path shares the review lock,
so a suggestion during a running review is refused rather than queued. The saved response
carries the adapter, model and device provenance and an explicit statement that a suggestion is
not evidence that the object is present. A backend that is unreachable, rejects the request or
answers unparseably is a sanitized 502 recorded in the server log: it is not the caller's
mistake, and an upstream message can name an internal endpoint, so neither is published.

## Perception adapters

Four adapters implement the same `Detector` protocol and feed the same reducer. A box thinner than the registration minimum is not a location this product can act on, so every adapter reports that object as `not_detected` for the frame rather than failing the run. **Reference tracking** is classical template correlation without learned weights. **RT-DETR + ByteTrack** proposes COCO-category boxes on CUDA, optionally disambiguated by the identity matcher below. **Cosmos grounding** (`vision/cosmos_detector.py`, backend `cosmos`) sends every sampled frame once per registered *category* together with that category's registration crops and asks for the box of every visible instance (accepted in unit or 0–1000 coordinates, near-duplicates merged). Which box belongs to which registered object is decided exactly as for RT-DETR: by the DINOv2 matcher when identity is enabled, otherwise one object per category with several candidates reported as `identity_ambiguous`. The sampling rate is clamped to `AGENTX_COSMOS_INDEX_FPS` (default 1) before the run is persisted, so freshness rules read the true cadence. Category requests for one frame run concurrently against the HTTP service. A malformed or retried-and-still-invalid answer yields no candidates, so the objects are `not_detected`; the model never supplies a score, a track ID or temporal reasoning. Run provenance records the adapter, model, endpoint, image edge, prompt hash and the identity configuration.

**SAM 2.1 temporal tracking** (`vision/sam2.py`, backend `sam2`) uses the Transformers video-session interface with a cached Small checkpoint. Each object is prompted with its normalized registration box on the original registration frame. Inference advances forward only; late registrations between sampled frames are processed at their original timestamp. Explicit frame indices remain monotonic after cache pruning. The adapter retains all conditioning outputs and 32 recent frames (the measured frame cache peaks at 34 entries, including conditioning frames), supports multiple object IDs, converts current masks to boxes, and marks nearly identical overlapping masks as ambiguous. Empty or sub-minimum masks produce no visible evidence.

The SAM2 presence guard (`vision/presence.py`) requires a minimum object-presence score, followed by either a higher presence score or DINOv2 similarity to the registration/eight most recent accepted views. Rejected and overlapping-ambiguous masks do not update the appearance bank. With identity disabled, the stricter presence score is required. The guard addresses observed occluder substitution but does not guarantee identity: identical objects and merged masks still fail in the first evaluation. Provenance records the exact weight checksum, runtime, dtype, cache, score semantics and guard thresholds. See [the measured results](benchmarks/INDEX.md).

On CUDA, `vision/masks.py` reduces full-resolution masks to integer bounding coordinates and overlap decisions on the GPU. Only these compact results and a batch of presence scores cross to the CPU. Pixel bounds, minimum box size, the 0.85 mask-IoU ambiguity threshold, presence filtering and retired-companion exclusions retain their original semantics. Integer overlap arithmetic avoids reduced-precision matrix products, and pair processing uses one temporary source-size mask at a time. CPU inference keeps the NumPy path. The [mask benchmark](../scripts/eval/benchmark_mask_geometry.py) checks exact geometry against that reference and measures this stage separately from end-to-end indexing.

## Identity matching

The optional `AGENTX_SAM2_DISTRACTOR_GUARD=true` adds at most four RT-DETR companion tracks on the first SAM2 registration frame. These are internal same-category alternatives, excluded from inventory and evidence. Mask ambiguity or bounding-box intersection covering 60% of the smaller same-category box quarantines the public identity for the rest of that run. After three accepted masks establish a size baseline, a mask bounding box more than four times the median of the eight most recent accepted areas also quarantines identity. This catches some merges with look-alikes that entered after registration; it can also abstain on a legitimate rapid scale change. A later user registration retires a matching internal track. The guard never automatically restores identity after a collision or size anomaly. Completed run provenance includes its policy and initialization boxes. See [the measured gains, unchanged reserved validation, and limitations](benchmarks/INDEX.md).

## Camera pose and what a region name means

The registration frame defines the coordinate system a user draws regions in, so a camera that is nudged, panned or zoomed changes what image coordinates mean. `vision/scene.py` estimates that change per sampled frame: ORB features outside the registered boxes, ratio-tested matches, a RANSAC partial-affine fit, and acceptance only with spatially distributed inliers in both frames and a low mean reprojection residual. A locally consistent patch still cannot assert a global transform. When the registration frame stops being matchable, the estimate is composed through the most recent supported view, at bounded depth and never backwards in time.

The decision the tracker makes is **not** whether to observe. A detection is evidence about a frame the user can replay, and a camera move does not make that box wrong; it makes the *region name* unsupported, because that name was drawn on the registration frame. So each observation records `scene_reference` (migration `0004`):

| `scene_reference` | Camera evidence | Stored box | Region |
| --- | --- | --- | --- |
| `registered` | at the registered pose, or no supported change assertion | current frame | read directly |
| `compensated` | moved, with a supported transform | current frame | read from the position mapped back into registration coordinates; none when that position leaves the registered view |
| `unavailable` | moved, with nothing relating the view to the registration frame | current frame | none |

Earlier releases only detected the change and then abstained permanently, which discarded every later observation in a recording whose camera moved once. That cost 283 of 300 visible samples on one benchmark clip. The current behaviour is measured in [camera recovery](benchmarks/INDEX.md#camera-movement-and-identity).

A grounded Cosmos review reads its points the same way, and it does not estimate anything itself.
Indexing already recorded a pose for every sampled frame of that run, with the registered objects
masked out and the whole forward history available, so the review looks those up by timestamp
(`Memory.camera_poses`, matched within half a sample interval) and maps the model's point into
registration coordinates before code names a region. The published coordinates stay in the frame
the model saw; a frame with no usable pose gets no region, is excluded from the region path, and
is counted in the summary. Without this the review borrows the user's region names for a different
coordinate frame and can report a movement that never happened.

An adapter may also use the pose. `vision/protocols.py` declares the optional `CameraAware` boundary; the reference matcher implements it and rewarps each registration template by the estimated transform before correlating, because a template is a picture taken from the registered pose. Trackers that carry their own memory through motion do not need it. None of this is SLAM: a supported affine fit is not a metric pose and does not prove a rigid planar scene. The policy and the per-run frame counts are recorded in run provenance.

RT-DETR proposes category boxes; an optional DINOv2 matcher (`vision/identity.py`, `AGENTX_IDENTITY_ENABLED=true`) decides which proposal is the registered object. Each registration crop is embedded once (CLS plus mean patch token, L2-normalized). Per frame, confident proposals of that category are embedded and assigned greedily by cosine similarity: below `AGENTX_IDENTITY_THRESHOLD` the object is `identity_unconfirmed`; a competing proposal within `AGENTX_IDENTITY_MARGIN` makes it `identity_ambiguous`; and when two registered objects match one proposal about equally, both become ambiguous and the proposal is spent, so this detected tie becomes uncertainty. The same matcher serves the RT-DETR and Cosmos adapters. This lifts the one-object-per-category restriction and can turn look-alike candidates into an explicit uncertain state; undetected identity switches remain possible. The similarity is stored as `identity_score` on the observation (migration `0003`) and the matcher's model, revision and thresholds are recorded in run provenance. A cosine similarity is a measurement, not a calibrated probability of identity.

## Live capture

A live capture is a `Video` with `live_status="recording"` whose source file grows.
`services/live.py` owns one fragmented H.264 writer per capture: each posted JPEG is decoded,
stamped with the server's monotonic clock on arrival (frames closer than 0.75 of the capture
interval are dropped), resized to at most `AGENTX_LIVE_MAX_EDGE` and muxed as its own fragment, so
every frame but the newest is readable while the file grows. `duration_ms` is the readable end.

While recording, the source-integrity check only requires the file to exist (there is no final
hash), answers carry a warning, and Cosmos review and evidence export are refused. The indexer's
follow mode decodes from the earliest registration, writes each pass immediately, polls every
250 ms and ends only after one full pass over the sealed file. Captures use a quarter-interval
sampling slack in both follow mode and rebuilds, so a rebuild of a sealed capture reproduces the
followed memory exactly (`tests/test_live_capture.py`). Sealing closes the writer, hashes the file,
creates the playback proxy and poster, and records `capture.sealed_at`; the run then stores the
sealed hash as `input_sha256`. Captures left open by a stopped process are sealed by the next
process's watcher; captures without frames for `AGENTX_LIVE_IDLE_SEAL_SECONDS` are sealed too.

## Jobs and failures

Index jobs are persisted before they execute. `POST /api/v1/runs/{id}/cancel` stops a queued run immediately or persists a running run as `cancelling` until the worker acknowledges it between inference calls. Already committed batches remain in the cancelled run for diagnosis; the unfinished run cannot answer questions. Cancellation is idempotent, cannot change a completed version, blocks edits until acknowledgement, and survives restart. A stopped version can be rebuilt as a fresh run. One worker processes them sequentially, writing batches of observations, events and progress in transactions. API reads use the committed watermark. An interrupted running job is failed on restart; queued jobs remain eligible. Rebuilding makes a new immutable run and preserves the failed record for diagnosis.

The deployment supports **one process**. The enqueue mutex is not a distributed lock. Introduce leases/atomic claims and a separate worker process before horizontal scaling. Provider timeouts and invalid outputs cause a visible fallback or failed review, not synthesized success.

## API examples

See authenticated `/docs` for the complete request schema. Normal operations:

```text
POST   /api/v1/videos                    Multipart video upload
POST   /api/v1/videos/{id}/objects       Register name, category, bounds and time
POST   /api/v1/videos/{id}/runs          Queue a new index
GET    /api/v1/runs/{id}                 Status, progress and model provenance
POST   /api/v1/runs/{id}/cancel          Persist a stop request for an active run
GET    /api/v1/runs/{id}/state?at_ms=7000
GET    /api/v1/runs/{id}/events?at_ms=7000
POST   /api/v1/runs/{id}/questions
GET    /api/v1/observations/{id}/frame   Original evidence frame
GET    /api/v1/videos/{id}/media         Authenticated, byte-range playback
POST   /api/v1/runs/{id}/review          Optional Cosmos interpretation
GET    /api/v1/runs/{id}/export          Portable metadata/evidence manifest
GET    /api/v1/questions/{id}/bundle    Offline answer report and original-frame ZIP
DELETE /api/v1/videos/{id}               Delete this video's data
```

Example question body:

```json
{"text":"Where is Red toolkit?","at_ms":7000,"use_provider":true,"use_skills":true}
```

`services/evidence_bundle.py` exports a single saved answer after revalidating its claims against the run and checking the source video hash. Its static HTML, JPEG frames, JSON and standalone integrity verifier work without the API. Cutoff context is labeled separately from observed evidence; the report never includes future frames. The exporter is serialized, bounded and temporary. See [the report contract](EVIDENCE_REPORTS.md).

`use_provider` selects the tool agent when a planner is configured; `object_id` pins the object and, on `/review`, switches Cosmos to grounded per-frame localization of that object.

A future robot adapter should consume structured state, evidence time and uncertainty. Image-region names are not navigation coordinates. World-frame transforms, calibration, live sensor age and motion-safety checks belong in a separate robotics integration.

## Schema evolution and deployment ownership

Alembic applies committed migrations at startup and through `agentx migrate`. Do not use `create_all` to bypass history or alter a published migration after adoption. Add a new revision for changes. SQLite is the tested default; validate migrations and job behavior separately before choosing PostgreSQL. Stop the single process for a consistent backup and store database plus media together.

Only upgrades are supported. The application enables SQLite foreign keys on every connection, and Alembic's batch `drop_column` rebuilds a table, so a SQLite downgrade of 0003–0006 cascade-deletes dependent runs, observations and answers. Restore a backup instead of downgrading.

HTTP response DTOs are validated Pydantic models and appear in OpenAPI. Frontend DTOs are currently explicit TypeScript interfaces. Integration tests cover this boundary; when API surface grows, introduce generated response-schema types before maintaining multiple consumers. Authentication is a single shared workspace token; tenant isolation and per-user authorization require a separate design.

## Visual review boundary

`Reviews` persists successful Cosmos interpretations in a separate `reviews` table (migration `0002`). They never enter `observations`, `events` or the causal reducer. Each saved response records the source video hash, cutoff, ordered frame timestamps and JPEG hashes, model label, backend/device provenance, generation attempts and the loaded `review-visual-evidence` Skill hash. Frame endpoints regenerate the exact bounded JPEG and reject a hash mismatch. Export schema `1.1` adds `reviews` while preserving existing fields.

`Providers.review` samples at most eight source frames at 1 FPS within the preceding seven seconds, longest edge 448 pixels by default. Two modes share that sampling:

- **Free-form** (no `object_id`): one multi-image request; the model returns `summary`, `evidence_frame_ids` and `uncertainty`. With `AGENTX_COSMOS_THINKING=true` it may reason inside `<think>` tags first; the parser strips the block and validates exactly one JSON object.
- **Grounded** (`object_id` supplied): one single-frame request per sampled frame, each carrying the object's registration crop as a reference image and returning `present`, a normalized center `x, y` and a note. The reference crop cannot be used before its registration time: earlier cutoffs return HTTP 422, and grounded review sampling starts no earlier than registration. Requests run concurrently against an HTTP service. The temporal logic lives in `domain/grounding.py`: zones are computed from the point with the run's region definitions, the last present frame and the later absent frames are named in code, and the cited frames are derived, not chosen by the model.

Reference-crop reviews can opt into `AGENTX_COSMOS_REFERENCE_GROUNDING=integer_1000` with `AGENTX_COSMOS_REFERENCE_MAX_EDGE=768`. This contract explicitly separates the identity crop from the source frame, requires integer coordinates from 0 to 1000 on each source-image axis, and converts by a fixed divisor of 1000 to the public 0–1 points. Decimal, string and boolean coordinates are rejected rather than guessed; absent coordinates must be null. Provenance records the actual image edge, model units, divisor and prompt hash so saved frame replay remains exact after configuration changes. Defaults retain the original normalized contract. Free-form reviews, grounding without a reference, and the separate Cosmos indexing adapter retain their existing configuration. The matched bottle comparison improved point localization but retained 3/11 false-visible absent requests; this remains a non-authoritative model interpretation. See [the reviewer results](benchmarks/INDEX.md#visual-reviewer-and-model-tuning).

The per-frame request is built by two public functions, `grounding_system` and
`grounding_messages` in `agents/providers.py`. Serving and the adapter's training data
(`scripts/training/build_grounding_sft.py`) both render through them, and a test pins their output to the
system-prompt hashes recorded by the deployed service, so a fine-tuned reviewer learns exactly the
request it will be sent. When the review endpoint serves a LoRA adapter, vLLM lists it in
`/v1/models` with its base checkpoint as `parent`; `Providers.served_model()` reads that entry once
(an unreachable service is asked again at most once a minute) and every review's provenance
records it as `served_model`, so an adapter's interpretation is never presented as the base
model's. See [the adapter and reviewer comparison](benchmarks/INDEX.md#visual-reviewer-and-model-tuning).

The grounded mode was introduced after multi-frame temporal errors on the controlled fixture. Public-video comparisons also found substantial single-frame localization errors: the base reviewer placed 270 of 366 development points correctly. Per-frame decomposition makes ordering inspectable; it does not establish visual truth. See [the reviewer results](benchmarks/INDEX.md#visual-reviewer-and-model-tuning). Both HTTP and native inference receive the same prompts. The native adapter lazily loads an existing local checkpoint, does not trust remote code, and serializes model loading/generation using one application-level review lock. A concurrent review receives HTTP 429. Use the HTTP service boundary (vLLM on the node) when hard process isolation or separate GPU scheduling is needed.

The parser accepts exactly one JSON object, optionally after a reasoning block or inside one Markdown fence. It rejects extra fields, trailing text, invalid/duplicate frame references and booleans/string IDs. One schema retry reuses the original evidence; invalid raw output is neither stored nor echoed. Valid references establish that a frame was supplied, **not that the model's visual assertion is true**. The UI exposes all sampled frames and, for grounded reviews, the per-frame verdicts.

Additional endpoints:

```text
GET /api/v1/runs/{id}/reviews           Saved visual reviews in chronological order
GET /api/v1/reviews/{id}/frames/{frame} Exact bounded JPEG supplied to that review
```
