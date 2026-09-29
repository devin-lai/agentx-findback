# What a FindBack answer means

Read this before wording an answer. Every rule below is enforced by the server in code; the
reply must not claim more than the server did.

## The cutoff

Every answer is "as of" a cutoff in video time (`as_of_ms`, printed as `cutoff`). Only
observations at or before the cutoff, in the chosen run and recording, can support it. Moving
the cutoff backwards asks what was known at that moment; later footage never leaks into an
earlier answer. An object cannot be recalled before the frame it was registered on.

## Status

| `status` | Meaning | How to say it |
| --- | --- | --- |
| `visible` | A recent sighting at the cutoff, with no later missing or ambiguous record. Freshness is bounded by the sampling rate. | "At 00:09.0 the Blue remote is visible in the Center area of the recording." |
| `last_seen` | Seen earlier, not at the cutoff. `evidence` is the last sighting; `current_zone` is empty. | "Last seen at 00:05.8 in the Right area; its position at 00:07.0 is not confirmed." |
| `unknown` | The camera view could not be trusted, the identity was ambiguous, or the original media is missing or changed. Old evidence can explain the past, never the present. | "The recording cannot confirm where it was at 00:07.0" plus the reason. |
| `not_observed` | No visible sighting at or before the cutoff. No location exists to report. | "The recording has no sighting of it up to 00:07.0." |
| `null` (script) | No single registered object matched the question, or several did. | Ask the user which registered object they mean, or to register it. |

`last_seen` is never `visible`. An object that disappeared may be hidden, removed, occluded or
simply missed by the detector; the recording does not say which, and neither may the reply.
Never name a destination, a container or a cause that was not observed.

## Reason codes (`reason`)

- `not_detected`: the latest sample had no detection.
- `observation_expired`: the last sighting is older than the freshness window.
- `identity_ambiguous`: several candidates matched; the identity is uncertain.
- `identity_unconfirmed`: a candidate did not match the registered appearance.
- `scene_changed`: the camera view changed in a way memory could not follow.
- `evidence_missing`, `evidence_changed`, `evidence_unavailable`: the original video file is
  gone, altered (hash mismatch) or unreadable, so no visual claim is supported.

## Intent and status are different things

`intent` is what was asked: `location` ("where is"), `last_seen` ("last seen"), or `history`
("what happened", "history", "timeline"). `status` is what memory supports. A location
question can correctly end with `last_seen`. `--intent` overrides the server's reading of the
question; do not use it to change the meaning of the user's words.

## History and events

`history` answers list the recorded changes up to the cutoff. Event kinds: `appeared`,
`moved` (confirmed by two consecutive samples in the new area), `reappeared`, `lost` and
`ambiguous`. The answer text names only `appeared`, `moved` and `reappeared` changes, each with
its own evidence frame (`other_evidence`). Sampling can miss short events.

## Areas and a moved camera

Areas (`zone`, `zone_name`) are the regions drawn for the recording, by default the Left,
Center and Right thirds. A box in no single region is "an unassigned area".
`scene_reference` tells how the evidence frame relates to the registration view: `registered`
(same view), `compensated` (the camera moved; the area was recovered in registration
coordinates, so it names the same place, not the same pixels), `unavailable` (the frame proves
the object was seen but not which registered area it was in).

## Runs are immutable

Building memory creates a new run with a frozen object inventory and region names. Answers
belong to one run (`run_id`). Registering another object, or renaming regions, does not change
an existing run: index again, and prefer a run marked `"current": true` by `videos`. Older runs
and their saved answers stay valid for what they covered.

## Who wrote the answer

- `planner: local` is the deterministic resolver. It matches registered names and categories
  and three intents; indirect descriptions such as "the item that disappeared" need the agent.
- `planner: agent` is the server's tool agent (for example NVIDIA Nemotron on vLLM, or any
  OpenAI-compatible endpoint). It chooses tools and an object. It never writes the location
  facts: the final text is rendered from verified memory records.
- Citations are re-validated in code. Every cited observation must belong to the run, the
  recording and the selected object, be visible, and precede the cutoff. Invalid citations are
  removed and a warning explains it. A planner failure falls back to the local resolver with a
  warning. Report warnings as they are.
- `skills` lists the server's own Skills that shaped the prompt, with SHA-256 prefixes. The full
  text behind each hash is published at `GET /api/v1/skills`.

## Model interpretations are not memory

Cosmos-Reason2 visual reviews are separate records marked `authoritative: false`. They can
describe what sampled frames show, but they never change observations, events or answers, and a
reply must not present one as an object's location. This skill does not request reviews.

## What a recording cannot tell

- Where anything is now in the real world. The answer describes the recording.
- Whether a disappeared object was moved, hidden, taken or missed by the detector.
- Who handled an object. FindBack does not identify or track people.
- Anything about objects that were never registered.
