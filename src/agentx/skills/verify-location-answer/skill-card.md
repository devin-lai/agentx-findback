# Skill Card: Verify Location Answer

Section names and field labels follow the NVIDIA Skill Card template (`NVIDIA/skills`
`docs/skill-cards.mdx` and the `skill-card-generator` template). This card was written by the
Skill's authors. It is not an NVIDIA-verified skill card.

## Description:

Check that an answer describes only supported visual facts within the requested time.

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

The FindBack tool agent reads this Skill before a location or history answer is returned; the deterministic resolver records it for provenance. It states the evidence scope, time bounds and status wording an answer must respect.

Loaded by `QueryWorkflow` in `AGENT_SKILLS` and `LOCAL_SKILLS`. Not loaded for typed selection, Cosmos review output (never authoritative) or indexing.

### Deployment Geography for Use:

Global.

## Requirements / Dependencies:

**Requires API Key or External Credential:** [Optional]

**Credential Type(s):** [API key]

Only the planner endpoint may need a key; the deterministic resolver needs none. Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate
keys as appropriate.

- The AgentX FindBack application (`src/agentx`), whose `SkillRegistry` reads the file.
- None beyond the application for the resolver; a planner endpoint for the agent.

## Known Risks and Mitigations:

Risk: A stored recording is presented as a real-world current position.
Mitigation: Code renders "This describes the recording, not its real-world current location" for visible answers and "not confirmed" for last sightings.

Risk: A model score is presented as calibrated certainty.
Mitigation: The Skill forbids percentages as certainty; answers carry no percentages.

Risk: Missing source media is ignored.
Mitigation: Code downgrades the status to unknown when the original recording is missing or changed.

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

**Output Type(s):** [Analysis]

**Output Format:** [Instructions only; shapes the agent's final `AgentStep`]

**Output Parameters:** [1D]

**Other Properties Related to Output:** [The published answer is rendered and scope-checked in code (`services/answering.py`).]

## Evaluation Tasks:

6 tasks in `evals/evals.json`: 3 cases in which the application must load
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

1.0.0 (source: frontmatter `version`). SHA-256 of `SKILL.md`: `8696dd59efafface8d68b64509dc2be88437a2d530bcc8cc636bfc15cac75c31`.
Detached signature: `skill.oms.sig`, ECDSA P-256, public key `skills/findback-skills.pub`, key
hint `0aa1a9e4bcac58f082cb852f099941346cabdb6c44c714408bddb064ed95a989`.

## Ethical Considerations:

The Skill exists to keep a model from claiming more than a recording shows. FindBack does not
identify people, infer theft or establish where an object is now, and a person should review the
cited evidence before acting on an answer. Report security issues privately as described in the
repository's `SECURITY.md`.
