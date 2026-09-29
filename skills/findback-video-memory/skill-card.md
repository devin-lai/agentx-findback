# Skill Card: FindBack Video Memory

Section names and field labels follow the NVIDIA Skill Card template
(`NVIDIA/skills` `docs/skill-cards.mdx` and the `skill-card-generator` template). This card was
written by the skill's authors. It is not an NVIDIA-verified skill card.

## Description:

Answers questions about registered objects in an uploaded fixed-camera recording through an
AgentX FindBack server: where an object was last seen before a cutoff time, what happened to it,
its history, and the original evidence frame behind the answer. It selects one recording;
searching for instances across an archive of videos is outside its scope.

This skill is for research and development only. It is part of a hackathon MVP; its measured
results come from generated fixtures and a small set of public clips, not from a production
deployment.

## Owner

AgentX FindBack team

## Third-Party Community Consideration

This skill is not owned or developed by NVIDIA. It was developed by the AgentX FindBack team for
this application and use case. It is team-authored and has **not** been verified, scanned,
evaluated, signed or published by NVIDIA. Its `skill.oms.sig` is signed with the team's own
ECDSA P-256 key (`skills/findback-skills.pub`), not with the NVIDIA agent root certificate.

### License/Terms of Use:

Apache-2.0 (`license` in the `SKILL.md` frontmatter; repository `LICENSE`).

## Use Case:

Developers and operators of an AgentX FindBack deployment, and the people they assist, use this
skill from a coding or chat agent (Claude Code, Codex, Cursor, OpenClaw or another harness that
loads Agent Skills) to ask where a registered object was last seen in a recording, what happened
to it before a given time, and to retrieve the cited original frame or an offline evidence
report. It is not for archive-wide search, live locations, people, whole-video summaries or
video editing.

### Deployment Geography for Use:

Global.

## Requirements / Dependencies:

**Requires API Key or External Credential:** [Optional]

**Credential Type(s):** [API key]

The FindBack access token (`FINDBACK_TOKEN`, the server's `AGENTX_API_TOKEN`) is a bearer
string. It is needed only when the server has one configured; a server without a token accepts
loopback clients only. Do not include secrets in prompts/logs/output; use least-privilege
credentials; rotate keys as appropriate.

- Python 3.10 or newer; `scripts/findback.py` uses the standard library only.
- An AgentX FindBack server (`agentx serve` from this repository) reachable at `FINDBACK_URL`,
  default `http://127.0.0.1:9000`. The default OpenCV reference backend needs no model weights.
  RT-DETR, DINOv2, SAM 2.1, Cosmos-Reason2 and a planner endpoint (for example NVIDIA Nemotron
  on vLLM, or any OpenAI-compatible endpoint) are optional server-side components.
- Network access: HTTP(S) to `FINDBACK_URL` only. The script sends no telemetry.

## Known Risks and Mitigations:

Risk: A last sighting in a recording is reported as the object's present location.
Mitigation: The server renders `last_seen` answers as "not confirmed" and every answer as a
statement about the recording; the skill forbids present-location claims and routes "right now"
questions without a recording away. Evals `fvm-011` and `fvm-101` test both paths.

Risk: The agent guesses where an unseen object went, or who moved it.
Mitigation: The skill and `references/answer-semantics.md` forbid naming unobserved
destinations, containers or causes; the server never infers them. Evals `fvm-004` and `fvm-009`.

Risk: The access token leaks into a transcript, a command line or a log.
Mitigation: The token is read from the environment only and never accepted as an argument. The
script never prints it, replaces it with `[REDACTED]` in all output, refuses HTTP redirects
(Python's `urllib` would forward the `Authorization` header to the redirect target), bypasses
proxies for loopback URLs and warns when a token would travel over plain HTTP to a remote host.
Covered by `tests/test_skill_package.py` and eval `fvm-010`.

Risk: Recordings can show people; the skill could be misused to identify or follow someone.
Mitigation: Identifying or tracking people is out of scope in the description, the body and the
server. Only objects the user names and locates are registered; `suggest` output is advisory.
Negative evals `fvm-102` and `fvm-104`.

Risk: The skill changes server state or writes files.
Mitigation: Upload, registration and indexing run only on an explicit request; `index --reuse`
avoids rebuilding memory that already exists; there is no delete command. Frames and reports are
written only to the paths passed with `--save-frame` and `--out`, atomically, and an existing
report is never overwritten without `--force`. GPU backends are used only when the user asks.

Risk: A server that is not trustworthy returns misleading or instruction-like text in answers.
Mitigation: Use the skill only with a FindBack server you operate. The script prints server data
as JSON fields and executes nothing it receives; the agent should treat those values as data.

Risk: Learned perception follows the wrong object with confidence (a documented limitation of
the tracking backends).
Mitigation: Every answer carries a replayable original frame and its time; inspect the cited
frame when an answer matters. The default reference backend is a transparent baseline.

Risk: An installed copy of the skill has been modified.
Mitigation: The directory carries a detached OpenSSF model-signing signature, `skill.oms.sig`,
over every file. Strict verification with `skills/findback-skills.pub` fails on any changed byte
or any unsigned addition.

## Reference(s):

- [Routing, required questions and safety boundaries](SKILL.md)
- [REST endpoints and fields](references/api.md)
- [Answer semantics: visible, last_seen, unknown, not_observed](references/answer-semantics.md)
- [Troubleshooting](references/troubleshooting.md)
- [Command-line client](scripts/findback.py)
- [Evaluation task set](evals/evals.json) and [benchmark protocol](BENCHMARK.md)
- AgentX FindBack repository: `README.md`, `docs/ARCHITECTURE.md` (memory contract) and
  `docs/EVIDENCE_REPORTS.md` (portable reports)
- [Agent Skills specification](https://agentskills.io/specification)
- [OpenSSF model signing](https://github.com/sigstore/model-transparency)

## Skill Output:

**Output Type(s):** [API Calls, Analysis, Files]

**Output Format:** [JSON on stdout from `scripts/findback.py`; the agent's reply in plain text or Markdown; JPEG evidence frames; ZIP evidence reports]

**Output Parameters:** [1D]

**Other Properties Related to Output:** [Answers quote the server's text rendered from verified memory, with status, evidence time, area and cutoff. For OpenClaw-style harnesses the reply ends with a standalone `MEDIA:<absolute path>` line when a frame was saved. Exit codes: 0 ok, 2 usage, 3 server or HTTP error, 4 timeout.]

## Evaluation Tasks:

18 development tasks in `evals/evals.json` (12 positive skill-activation cases and 6 negative
cases, `expected_skill: null`, `should_trigger: false`) and two held-out sets of 8 (5 positive,
3 negative each), each frozen before the session that first used it. `BENCHMARK.md` gives the
protocol, the commands and every run.

## Evaluation Metrics Used:

Reported benchmark dimensions, as defined by NVIDIA SkillEvaluator Tier 3:

- Security: checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access.
- Correctness: checks whether the agent follows the expected workflow and produces the correct final output.
- Discoverability: checks whether the agent loads the skill when relevant and avoids using it when irrelevant.
- Effectiveness: checks whether the agent performs measurably better with the skill than without it.
- Efficiency: checks whether the agent uses fewer tokens and avoids redundant work.

## Evaluation Results:

On three small held-out sets, two coding agents each improved from 1/15 to 15/15 positive answers with this Skill. Their development results were 6/18 to 17/18 and 8/18 to 16/18. One agent activated the Skill for one of 15 negative prompts, but still declined to claim a live location. A smaller local model improved from 2/16 to 5/16 positive answers and activated on 7/9 negative prompts. See `BENCHMARK.md` for the limits of these generated-fixture measurements. No broad user or tracking accuracy claim follows from them.

## Testing Completed:

**[ ] Agent Red-Teaming**

**[ ] Network Security**

**[ ] Product Security**

Automated tests in the repository cover argument parsing, token redaction, redirect refusal and
an end-to-end run against a local FindBack server (`tests/test_skill_package.py`). No red-team,
network-security or product-security review has been performed.

## Skill Version(s):

1.1.3 (source: frontmatter `metadata.version`). Detached signature: `skill.oms.sig`, ECDSA P-256,
public key `skills/findback-skills.pub`, key hint (SHA-256 of the PEM public key)
`0aa1a9e4bcac58f082cb852f099941346cabdb6c44c714408bddb064ed95a989`.

## Ethical Considerations:

Recordings of homes and workplaces can show people and private spaces. Record and upload only
footage you are entitled to use, and keep the server on a trusted network with an access token.
The skill reports what a recording shows at a stated time, with the evidence that supports it; it
does not decide where something is now, why it moved, or who moved it, and a person should review
the cited frame before acting on an answer. Report security issues privately as described in the
repository's `SECURITY.md`.
