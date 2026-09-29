---
name: findback-video-memory
description: Answers questions about registered objects in one uploaded fixed-camera recording via an AgentX FindBack server - where an object was last seen before a cutoff time, what happened to it, whether it is still there at the end of the recording, and the evidence frame behind the answer. Use for "where was the remote last seen in the video", "what happened to the scissors", "is it still on the desk in the recording", "show me the evidence frame", "the toolkit's history before 00:07", or FindBack errors such as HTTP 401. Not for searching across an archive of videos for any instance or last appearance; use an archive-search skill for that. Also not for a live location with no recording to check, identifying or tracking people, summarizing whole videos, guessing where an unseen object went, or editing video.
license: Apache-2.0
compatibility: Requires Python 3.10+ (standard library only) and HTTP access to an AgentX FindBack server. Set FINDBACK_URL, and FINDBACK_TOKEN when the server has an access token.
metadata:
  version: "1.1.3"
  author: AgentX FindBack team
  tags: video-memory object-history evidence
---

# FindBack video memory

## Purpose

Answers "where was it last seen?" about objects in an uploaded fixed-camera recording, from an AgentX FindBack server's timestamped object memory. The server renders every location claim from verified observations and re-validates citations in code. This skill routes the request, asks for missing parameters and reports the answer with its evidence. It adds no claims of its own.

## Prerequisites

- Python 3.10+. `FINDBACK_URL` selects the server (default `http://127.0.0.1:9000`). `FINDBACK_TOKEN`, when the server needs one, is read from the environment only.
- Run `python3 scripts/findback.py <command>` from this directory. It prints one JSON document on stdout and hints on stderr. Exit codes: 0 ok, 2 usage, 3 server or HTTP error, 4 timeout.

| Script | Purpose | Arguments |
| --- | --- | --- |
| `scripts/findback.py` | FindBack REST client (standard library only) | `<command> [options]`; `--help` on any command |

## Required questions

Ask the user when any of these is missing. Never guess them.

1. **Which recording?** Its exact title as listed by `videos` — a title that only starts the same way is a different recording — or a file the user named for `upload`.
2. **Which object, and how to recognize it?** A registered name from `videos`. If it is not registered: a name, the time of a clear frame, and where it is in that frame (for `register --box`).
3. **As of when?** The cutoff in video time: 00:07 becomes `--at-ms 7000`. Use `--at-end` only when the user asks about the end of the recording or about "now"; for "now", say that the recording cannot show the present.

## Routing

| User intent | Command or reference |
| --- | --- |
| Is the server up, what can it do? | `health` |
| Which recordings, objects and runs exist? | `videos`, `video <video_id>` |
| Add a recording the user named / try without footage | `upload <file>` / `demo` (a generated, labeled fixture; say so) |
| Which objects could be registered? | `suggest <video_id> --at-ms N` (advisory; registers nothing) |
| Register an object the user named | `register <video_id> --name "..." --at-ms N --box x1,y1,x2,y2` (0-1; `--pixels` for pixels) |
| Build memory | `index <video_id> --reuse --wait` |
| Where was X, where was X last seen, what happened to X | `ask --video "<exact title>" "<question>" --at-ms N --save-frame <dir>` (or `ask <run_id> ...` for one specific run) |
| Show a cited frame again / export a verifiable report | `frame <observation_id> --out <dir>` / `report <question_id> --out findback-evidence.zip` |
| What `visible`, `last_seen`, `unknown`, `not_observed` mean | [references/answer-semantics.md](references/answer-semantics.md) |
| Endpoints and fields | [references/api.md](references/api.md) |

## Instructions

1. Run `health`. If the server is unreachable or answers 401, stop and see Troubleshooting.
2. Run `videos` and find the recording whose title is exactly the one the user means. If none or several could be meant, ask.
3. If it has no run with `"status": "complete"` and `"current": true`, and the user wants one, register their objects if needed and run `index <video_id> --reuse --wait`.
4. Run `ask --video "<exact title>"` with a short question about the object by its registered name, the cutoff in `--at-ms` (not in the question) and `--save-frame`. Save frames in the agent workspace or another output directory **outside this Skill directory**; writing into the Skill changes its signed contents.
5. Reply using the output contract. Keep every server warning.

## Output contract

- The recording's title (`recording`) and the `answer` text as returned. The server wrote it from verified memory.
- Say that the answer describes the recording ("in the recording", "at 00:09 of the video"), never the present.
- `status`: `visible`, `last_seen`, `unknown` or `not_observed`. `null` means no single registered object matched: ask which one the user means, or how to recognize it for registration.
- The evidence time and area (`evidence.time`, `evidence.zone_name`), the cutoff (`cutoff`), the saved frame path (`evidence_frame`) and any `warnings`.
- For OpenClaw-style harnesses, when a frame was saved, copy the `media` value from the `ask` JSON verbatim as the **last line of the final reply**: `MEDIA:/absolute/path/findback-evidence-<id>.jpg`. No code fence, bold or other text on that line. `MEDIA:` is a reply marker for the harness, **not a shell command**: never pass it to `exec`. Do not open the JPEG with a text-reading tool.

## Examples

User: "In the controlled desk fixture, where was the Red toolkit last seen as of 00:07?" Run `python3 scripts/findback.py ask --video "Controlled desk fixture" "Where was the Red toolkit last seen?" --at-ms 7000 --save-frame /agent/workspace/evidence`, then reply: "In the recording "Controlled desk fixture", the last supported observation of Red toolkit at or before 00:07.0 is at 00:05.8, in Right area. Its position at the requested time is not confirmed (status `last_seen`, cutoff 00:07.0). Evidence frame: /agent/workspace/evidence/findback-evidence-<id>.jpg", followed by the `MEDIA:` line when the harness uses one.

User: "Where is my phone right now?" This skill does not apply: a recording cannot show where the phone is now. Offer to find when it was last seen in a named recording.

## Safety boundaries

- This Skill answers from one selected FindBack recording and its registered objects. A search across many archived videos belongs to an archive-search Skill, even if the wording asks where an object was "last seen".
- A recording shows the past. It cannot prove where anything is now. Answer "now" or "currently" questions about a recording with what it shows at its end, and say so. Without a recording, this skill cannot answer them at all.
- `last_seen` is not `visible`. Give the last sighting's time and area and say the position at the cutoff is not confirmed. Never guess where an unseen object went or why (a drawer, a bag, someone took it).
- Cite the status, the evidence time and the cutoff in every answer.
- Never print, echo, log or derive anything from `FINDBACK_TOKEN` - not its value, a prefix or suffix, or its length - never put it on a command line, and never ask the user to paste it into the chat. For HTTP 401, tell the user to set it in the agent's environment and check it in their own terminal.
- Register only objects the user named and located. `suggest` output is advisory. Never identify, name or track people.
- Uploading, registering and indexing change the server: do them only when the user asks. With a read-and-ask agent token they return HTTP 403; tell the user instead of retrying.
- Cosmos reviews are model interpretations, not memory. Never present one as an object's location.
- Reuse completed runs and saved answers instead of indexing again. Do not choose the `rtdetr`, `cosmos` or `sam2` backends, or `suggest --backend cosmos`, unless the user asks: they run GPU models.
- Do not change the server's configuration or `.env`, delete recordings, or edit video.

## Limitations

- Uploaded fixed-camera recordings only; sampling can miss short events. Occlusion, removal and detector failure look the same.
- A tracker can follow the wrong object with confidence. Inspect the cited frame when an answer matters.
- Without a planner on the server, only registered names and categories resolve; indirect descriptions ("the item that disappeared") need the server's agent.

## Troubleshooting

- `Cannot reach FindBack`: check `FINDBACK_URL` and that `agentx serve` is running.
- HTTP 401: set `FINDBACK_TOKEN` in the agent's environment, never in the chat. HTTP 403: the token is a read-and-ask agent token and the action needs the operator. HTTP 422 "That time has not been processed": use a smaller `--at-ms` or `--at-end`.
- Failed runs, planner fallback warnings and other errors: [references/troubleshooting.md](references/troubleshooting.md).
