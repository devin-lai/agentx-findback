# Skill Card: Answer With Memory Tools

Section names and field labels follow the NVIDIA Skill Card template (`NVIDIA/skills`
`docs/skill-cards.mdx` and the `skill-card-generator` template). This card was written by the
Skill's authors. It is not an NVIDIA-verified skill card.

## Description:

Drive the FindBack tool agent. Resolve the object, read memory state and history through tools, cite retrieved observation IDs, and state uncertainty before finishing.

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

The FindBack tool agent, a planner model such as NVIDIA Nemotron 3.5 Lightning served by vLLM or any OpenAI-compatible JSON-mode endpoint, reads this Skill when it answers a direct location, last-seen or history request about a registered object, or a request about an object the user selected explicitly.

Loaded by `QueryWorkflow` (`AGENT_SKILLS` in `src/agentx/agents/workflow.py`) when a planner is configured, `use_provider` is true and the question names a registered object or the user selected one. Not loaded for the deterministic resolver, typed selection, visual review or indexing.

### Deployment Geography for Use:

Global.

## Requirements / Dependencies:

**Requires API Key or External Credential:** [Optional]

**Credential Type(s):** [API key]

Only the planner endpoint may need a key (`AGENTX_PLANNER_API_KEY`); a local vLLM service needs none. Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate
keys as appropriate.

- The AgentX FindBack application (`src/agentx`), whose `SkillRegistry` reads the file.
- A planner endpoint set with `AGENTX_PLANNER_BASE_URL` and `AGENTX_PLANNER_MODEL`.

## Known Risks and Mitigations:

Risk: The planner cites observation IDs it never retrieved, or answers about a different object.
Mitigation: Code re-validates every citation and removes invalid ones with a warning; an answer about another object is replaced by the selected object's memory. The Skills ablation recorded no wrong-object answer in 504.

Risk: The planner turns a past sighting into a present location.
Mitigation: Status wording is rendered by code (`services/answering.py`); `last_seen` answers always say the position is not confirmed.

Risk: Text in a question tries to steer the tool loop.
Mitigation: The action space is five code tools; unknown tools and keys are ignored, the step budget is bounded and a failed loop falls back to the deterministic resolver with a visible warning.

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

**Output Format:** [JSON `AgentStep`: one tool call, or a final answer with `evidence_observation_ids`]

**Output Parameters:** [1D]

**Other Properties Related to Output:** [Code re-validates every cited observation against the run, the object and the cutoff, and renders the published answer from verified memory records.]

## Evaluation Tasks:

9 tasks in `evals/evals.json`: 4 cases in which the application must load
this Skill and 5 negative cases in which it must not
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

2 (source: frontmatter `version`). SHA-256 of `SKILL.md`: `57fc9d2218d33b29dd94397e675200708d6256b848848df804a338d69f70dde3`.
Detached signature: `skill.oms.sig`, ECDSA P-256, public key `skills/findback-skills.pub`, key
hint `0aa1a9e4bcac58f082cb852f099941346cabdb6c44c714408bddb064ed95a989`.

## Ethical Considerations:

The Skill exists to keep a model from claiming more than a recording shows. FindBack does not
identify people, infer theft or establish where an object is now, and a person should review the
cited evidence before acting on an answer. Report security issues privately as described in the
repository's `SECURITY.md`.
