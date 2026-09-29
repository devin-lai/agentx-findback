# Skill Card: Compile Object Selection

Section names and field labels follow the NVIDIA Skill Card template (`NVIDIA/skills`
`docs/skill-cards.mdx` and the `skill-card-generator` template). This card was written by the
Skill's authors. It is not an NVIDIA-verified skill card.

## Description:

Translate an object description into complete, causal selection criteria.

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

The FindBack tool agent reads this Skill when a question describes an object indirectly ("the item that disappeared", "the object never seen on the right") and must be translated into typed selection criteria that code evaluates over the frozen inventory.

Loaded by `QueryWorkflow` (`SELECTION_SKILLS`) when a planner is configured, `use_provider` is true, no object is selected and the question does not name a registered object exactly. Not loaded for exact names, explicit selections, the deterministic resolver, review or indexing.

### Deployment Geography for Use:

Global.

## Requirements / Dependencies:

**Requires API Key or External Credential:** [Optional]

**Credential Type(s):** [API key]

Only the planner endpoint may need a key (`AGENTX_PLANNER_API_KEY`). Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate
keys as appropriate.

- The AgentX FindBack application (`src/agentx`), whose `SkillRegistry` reads the file.
- A planner endpoint set with `AGENTX_PLANNER_BASE_URL` and `AGENTX_PLANNER_MODEL`. Typed selection sends `thinking_token_budget`, which vLLM accepts only with a reasoning parser; `agentx doctor --probe` checks it.

## Known Risks and Mitigations:

Risk: Instructions embedded in a question become selection criteria.
Mitigation: Measured: held-out cases 11 and 12 of the Skills ablation ended in refusals, not false locations, in both planner arms. Explicit selection bypasses compilation entirely.

Risk: The model relaxes or invents criteria to force a unique match.
Mitigation: The term validator rejects invented qualifiers and the plan is re-evaluated in code; code never falls back to a bare name when a named qualifier fails.

Risk: A missing detection is read as proof of absence or of a cause.
Mitigation: The Skill states that missing detections prove neither; code evaluates only recorded states and events.

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

**Output Format:** [JSON `SelectionPlan` of typed filters]

**Output Parameters:** [1D]

**Other Properties Related to Output:** [Code evaluates the plan over the frozen inventory and decides whether zero, one or several objects match; zero or several produce a refusal.]

## Evaluation Tasks:

8 tasks in `evals/evals.json`: 4 cases in which the application must load
this Skill and 4 negative cases in which it must not
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

1.0.0 (source: frontmatter `version`). SHA-256 of `SKILL.md`: `7b8d6be2dbdb682581bb18c63ccdcbb06c4212b856f76e9f94428c2a1744e067`.
Detached signature: `skill.oms.sig`, ECDSA P-256, public key `skills/findback-skills.pub`, key
hint `0aa1a9e4bcac58f082cb852f099941346cabdb6c44c714408bddb064ed95a989`.

## Ethical Considerations:

The Skill exists to keep a model from claiming more than a recording shows. FindBack does not
identify people, infer theft or establish where an object is now, and a person should review the
cited evidence before acting on an answer. Report security issues privately as described in the
repository's `SECURITY.md`.
