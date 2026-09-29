# Skill Card: Observe Object Events

Section names and field labels follow the NVIDIA Skill Card template (`NVIDIA/skills`
`docs/skill-cards.mdx` and the `skill-card-generator` template). This card was written by the
Skill's authors. It is not an NVIDIA-verified skill card.

## Description:

Build causal object events from registered visual observations.

This skill is for research and development only. It is an internal prompt contract of the
AgentX FindBack application, loaded verbatim into one step's system prompt; it is not meant to
be installed into a coding agent. The portable, installable skill is
`skills/findback-video-memory`.

## Owner

AgentX FindBack team

## Third-Party Community Consideration

This skill is not owned or developed by NVIDIA. It was developed by the AgentX FindBack team for
this application and use case. It is team-authored and has **not** been verified, scanned,
evaluated, signed or published by NVIDIA. Its `skill.oms.sig` is signed with the team's own
ECDSA P-256 key (`skills/findback-skills.pub`), not with the NVIDIA agent root certificate.

### License/Terms of Use:

Apache-2.0 (repository `LICENSE`). The `SKILL.md` frontmatter has no `license` field and keeps a
top-level `version` field that the agentskills.io reference validator rejects. Neither is
changed, because saved answers and benchmark receipts cite the SHA-256 of the exact bytes.

## Use Case:

Documents the causal event contract of every FindBack index run: how sampled observations become appeared, moved, lost, reappeared and ambiguous events. Its name and SHA-256 are recorded in each run's provenance.

Loaded by `Indexer` (`src/agentx/services/indexing.py`) for every index run; only its trace is stored. No model receives its text: the event reducer is code (`src/agentx/domain/temporal.py`). Never loaded at query or review time.

### Deployment Geography for Use:

Global.

## Requirements / Dependencies:

**Requires API Key or External Credential:** [No]

**Credential Type(s):** [None]

Indexing with the reference backend needs no credential. Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate
keys as appropriate.

- The AgentX FindBack application (`src/agentx`), whose `SkillRegistry` reads the file.
- The AgentX FindBack application; any perception backend.

## Known Risks and Mitigations:

Risk: The written contract and the reducer diverge.
Mitigation: The reducer is covered by `tests/test_temporal.py` and `tests/test_camera_recovery.py`; each run records the exact Skill hash it was built with.

Risk: A camera change is reported as object movement or disappearance.
Mitigation: Camera changes keep the observation and its box and name a region only while the frame can be related to the registration view (version 1.1.0 of this text).

Risk: The file is edited and answers no longer match the text they cite.
Mitigation: Every answer, review or run stores the SHA-256 of the exact text used, and
`GET /api/v1/skills` serves the current text and hash. The directory also carries a detached
signature, `skill.oms.sig`.

## Reference(s):

- [The Skill text](SKILL.md)
- [FindBack Skills catalog: triggers, negative triggers, inputs and outputs](../README.md)
- [Evaluation task set](evals/evals.json) and [benchmark record](BENCHMARK.md)
- `docs/ARCHITECTURE.md` (memory contract and agent design)
- `docs/benchmarks/INDEX.md`

## Skill Output:

**Output Type(s):** [Other [provenance record]]

**Output Format:** [Run provenance entry `{name, sha256, purpose}`]

**Output Parameters:** [1D]

**Other Properties Related to Output:** [Events and observations are produced by code; the Skill text is the written contract and audit record.]

## Evaluation Tasks:

5 tasks in `evals/evals.json`: 2 cases in which the application must load
this Skill and 3 negative cases in which it must not
(`expected_skill: null`). `tests/test_skill_package.py` executes the routing of every case against
the generated controlled fixture on each test run. Assertions about model output need a
configured model service.

## Evaluation Metrics Used:

Routing: whether the saved trace names this Skill exactly when `should_trigger` is true. Model
behavior: the Skills ablation's pass rate, wrong-object answers, fallbacks and guard counts
(`docs/benchmarks/INDEX.md`).

## Evaluation Results:

See `BENCHMARK.md`, which quotes only published measurements.

## Skill Version(s):

1.1.0 (source: frontmatter `version`). SHA-256 of `SKILL.md`: `e7e3fc0e89a4f50f9e5c3875878be42bfe744442744146e421083b30591205be`.
Detached signature: `skill.oms.sig`, ECDSA P-256, public key `skills/findback-skills.pub`, key
hint `0aa1a9e4bcac58f082cb852f099941346cabdb6c44c714408bddb064ed95a989`.

## Ethical Considerations:

The Skill exists to keep a model from claiming more than a recording shows. FindBack does not
identify people, infer theft or establish where an object is now, and a person should review the
cited evidence before acting on an answer. Report security issues privately as described in the
repository's `SECURITY.md`.
