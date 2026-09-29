# FindBack REST API used by this skill

Server version 0.1.0. Paths are relative to `FINDBACK_URL`. JSON in, JSON out, except where a
binary response is named. This lists only what `scripts/findback.py` calls; the server's full
OpenAPI document is at `/docs` (it needs the token when one is configured).

## Authentication

- With `AGENTX_API_TOKEN` set on the server, every `/api/v1/*` request needs
  `Authorization: Bearer <token>`. The script sends `FINDBACK_TOKEN` this way and never prints it.
  A missing or wrong token returns **401** `{"detail": "Sign in with the configured access token."}`.
- Without a server token, only loopback clients are accepted; others get **503**
  `{"detail": "Configure AGENTX_API_TOKEN before network access."}`.
- `/api/health` and `/api/auth/status` need no token.
- The script refuses HTTP redirects so the token cannot be forwarded to another host.

## Errors

`{"detail": "<message>"}` for 401, 404, 409, 413, 415, 422, 429, 502 and 503. Request validation
errors are **422** with `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}`. The server
maps invalid values (`ValueError`) to 422 and unknown IDs (`LookupError`) to 404. The script turns
any HTTP error into exit code 3 with `error.http_status` and `error.message`.

## Endpoints

| Command | Request | Success response |
| --- | --- | --- |
| `health` | `GET /api/health` | `{status, product, version}` |
| `health` | `GET /api/auth/status` | `{required}` |
| `health` | `GET /api/v1/capabilities` | `planner`, `planner_model`, `cosmos`, `cosmos_backend`, `cosmos_model`, `identity`, `sam2`, `rtdetr_installed`, `discovery` (`{rtdetr, cosmos}`), `reference`, `demo_scenes`, `skills`, `max_upload_mb`, `max_video_seconds`, `scope` |
| `videos` | `GET /api/v1/videos` | `VideoResponse[]`, newest first |
| `video` | `GET /api/v1/videos/{video_id}` | `VideoResponse` |
| `upload` | `POST /api/v1/videos`, multipart field `file`; `.mp4 .mov .mkv .webm .avi` | **201** `VideoResponse`. 413 over the size limit, 415 wrong type, 422 empty or undecodable |
| `demo` | `POST /api/v1/demo?scene=fixed\|bumped` | **201** `VideoResponse` with `is_fixture: true` and two registered objects |
| `suggest` | `POST /api/v1/videos/{video_id}/suggestions` `{at_ms, backend: "rtdetr"\|"cosmos"}` | `{at_ms, backend, proposals: [{suggested_name, label, box, score}], provenance, elapsed_seconds, limits}`. 503 backend not available, 502 backend failed, 409 busy |
| `register` | `POST /api/v1/videos/{video_id}/objects` `{name, label, at_ms, box: {x1, y1, x2, y2}}` | **201** `{id, name, label, registered_at_ms, box, reference_url}` |
| `index` | `POST /api/v1/videos/{video_id}/runs` `{backend, sample_fps?}` | **202** `RunResponse` |
| `index --wait`, `run` | `GET /api/v1/runs/{run_id}` | `RunResponse` |
| `ask` | `POST /api/v1/runs/{run_id}/questions` (`Question`) | `AnswerResponse` |
| `ask --video "<title>"` | `GET /api/v1/videos`, then the recording's latest complete run that covers its current objects | Matches the complete title, ignoring case and repeated whitespace. Balanced outer quotes passed literally by an agent are tolerated. Exit 2 when no title matches (the message lists titles), when several do, or when the recording has no such run. A shared prefix never matches. |
| `ask` (zone names) | `GET /api/v1/runs/{run_id}` | `RunResponse.regions` |
| `ask --save-frame`, `frame` | `GET /api/v1/observations/{observation_id}/frame` | `image/jpeg`, header `X-Frame-Time-Ms`. 422 when the observation is not visible, 404 when original media is missing |
| `report` | `GET /api/v1/questions/{question_id}/bundle` | `application/zip`. 429 busy (`Retry-After`), 413 over the report limit |

## Request bodies

- **Register:** `name` 1-80 characters, unique in the recording (case-insensitive); `label`
  defaults to `custom` (a detection category such as `remote` or `cup` enables the learned
  detector); `at_ms` must be inside the video; `box` is **normalized** to the source frame, each
  value 0-1 with `x2 - x1` and `y2 - y1` at least 0.005, and the crop at least 8 pixels. At most
  10 objects per video. Registration is refused while an index run is active.
- **Index:** `backend` is `reference` (OpenCV template matching, always available), `rtdetr`,
  `cosmos` or `sam2` (each needs server-side models); `sample_fps` 1-15, default 5;
  `match_threshold` 0.5-0.99, default 0.82 (not exposed by the script). Refused with 422 while
  another run is queued or running for the video, when no object is registered, or when the
  backend is not configured.
- **Question:** `text` 1-1000 characters; `at_ms` the cutoff, 0 up to the indexed range,
  omitted means the end of the run; `object_id` selects a registered object explicitly;
  `intent` forces `location`, `last_seen` or `history`; `use_provider: false` forces the local
  resolver; `use_skills: false` withholds the server's Skills (an ablation switch).

## Response models

- **VideoResponse:** `id, title, original_name, sha256, duration_ms, width, height, fps,
  regions[{id, name, box}], is_fixture, created_at, media_url, poster_url, objects[], runs[]`.
- **RunResponse:** `id, video_id, backend, status` (`queued`, `running`, `cancelling`,
  `cancelled`, `complete`, `failed`), `processed_ms, observation_count, elapsed_seconds, config,
  provenance` (includes the `observe-object-events` Skill hash), `error, created_at, object_ids,
  regions`.
- **AnswerResponse:** `id` (the question ID), `run_id, question, intent, as_of_ms` (the cutoff),
  `answer, states[ObjectMemory], events[], evidence[EvidenceRef], skills[{name, sha256,
  purpose}], tools[], planner` (`local` or `agent`), `planner_model, warnings[], elapsed_seconds`.
- **ObjectMemory:** `object_id, name, status, as_of_ms, last_observed_ms, zone, current_zone,
  reason, evidence`.
- **EvidenceRef:** `observation_id, video_id, run_id, at_ms, box, zone, scene_reference`
  (`registered`, `compensated`, `unavailable`), `scene_transform, frame_url, media_url`.
- **Event:** `id, object_id, name, at_ms, kind` (`appeared`, `moved`, `reappeared`, `lost`,
  `ambiguous`), `zone, previous_zone, reason, observation_id`.

## What the script prints for `ask`

`recording` (the recording's title), `question_id, run_id, question, intent, cutoff_ms, cutoff, answer, status, object, reason,
evidence{observation_id, at_ms, time, zone, zone_name, scene_reference}, warnings, planner,
planner_model, skills["name@sha256-prefix"], tools["name: result"], elapsed_seconds`, plus
`other_evidence`, `events` and `states` when present, and `evidence_frame` and `media` after
`--save-frame`. `--full` prints the server's `AnswerResponse` unchanged.
