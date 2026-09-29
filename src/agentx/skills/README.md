# FindBack Agent Skills

Six versioned instruction files that constrain how a model may participate in an
evidence-backed memory. Each is a plain `SKILL.md` loaded verbatim into the prompt of one
step, hashed, and recorded in the answer's trace.

These Skills are **safety and scope contracts, not capability prompts**. None of them can
retrieve an observation, choose an object, compute a zone or write an answer: application
code does all of that. A Skill decides what the model is permitted to assert on the way.

## How a Skill reaches the model

`SkillRegistry.load(name)` reads `skills/<name>/SKILL.md` and returns the body with a
`SkillTrace(name, sha256, purpose)`. `QueryWorkflow` selects the active set for the step,
appends the bodies to the system prompt, and stores every trace on the saved answer, so a
reader can recompute the hash of the exact text that produced it.

`GET /api/v1/skills` publishes the whole catalog — name, declared version, description,
purpose, SHA-256 and exact body — so a hash cited by a saved answer can be checked against the
instructions the running deployment actually holds.

`Question.use_skills=false` loads none; that is the ablation arm measured in
[the Skills ablation](../../../docs/benchmarks/INDEX.md).

| Step | Active Skills |
| --- | --- |
| Tool agent answering a direct request | `retrieve-object-history`, `verify-location-answer`, `answer-with-memory-tools` |
| Tool agent compiling an indirect description | `compile-object-selection` |
| Deterministic resolver (no planner) | `retrieve-object-history`, `verify-location-answer` (recorded for provenance; the resolver's behavior does not depend on them) |
| Cosmos visual review | `review-visual-evidence` |
| Indexing | `observe-object-events` |

## The six Skills

| Skill | Governs | Triggers | Does **not** apply to |
| --- | --- | --- | --- |
| `answer-with-memory-tools` | The bounded tool loop: which tool to call, what may be cited, when to stop | A location, last-seen or history question about a registered object | Choosing an object from an indirect description; any visual judgement |
| `compile-object-selection` | Translating a description into typed criteria evaluated by code | "the item that disappeared", "the object never seen on the right" | A question that names an object exactly; retrieval; answer text |
| `retrieve-object-history` | Resolving a name against the catalog and honoring the cutoff | Any question naming or implying one registered object | Objects that were never registered |
| `verify-location-answer` | The final check: evidence scope, time bounds, status wording | Before any location or history answer is returned | Visual review output, which is never authoritative |
| `review-visual-evidence` | What a Cosmos review may claim from sampled frames | A visual question about frames near a cutoff | Authoritative memory; registered identity |
| `observe-object-events` | Turning observations into causal events during indexing | Every index run | Query time |

### Negative triggers

These are the cases where loading a Skill would be wrong, and the workflow does not:

- **Do not** load `compile-object-selection` when the user selected an object explicitly or
  typed its exact registered name — the selection is already decided, and re-deriving it
  invites the model to overrule the user.
- **Do not** load `answer-with-memory-tools` for the deterministic resolver. It describes a
  tool loop that the resolver does not run.
- **Do not** load `review-visual-evidence` on the answering path. It permits interpretation
  of pixels, which must never reach a published location claim.
- **Do not** treat any Skill as a capability grant. A Skill that says "cite observation IDs"
  does not make the model able to invent one that passes code re-validation.

## Inputs and outputs

A Skill body is appended to a system prompt; it receives no arguments. The step around it has
a typed contract:

| Step | Input | Output validated by code |
| --- | --- | --- |
| Tool loop | Question text, cutoff, region names, short object IDs, tool results | `AgentStep`: one tool call, or a final answer whose `evidence_observation_ids` are re-validated against the run, the object and the cutoff |
| Selection | Question text, inventory, region names, cutoff | `SelectionPlan`: typed filters evaluated over the frozen inventory; zero, one or several matches decided in code |
| Review | Bounded JPEG frames and their timestamps | `Review` or per-frame `FrameReport`; stored as a non-authoritative interpretation |

Identifiers given to the model are per-question references (`object_1`), not database IDs;
real IDs are restored in the saved trace and evidence.

## Prerequisites

Any OpenAI-compatible chat endpoint with JSON mode, set through `AGENTX_PLANNER_BASE_URL` and
`AGENTX_PLANNER_MODEL`. The validated deployment serves NVIDIA Nemotron 3.5 Lightning with
vLLM on the DGX Spark node. Typed selection additionally sends `thinking_token_budget`, which
vLLM rejects unless it was started with a reasoning parser — `agentx doctor --probe` calls both
request shapes and reports which one an endpoint actually accepts.

With no planner configured, the deterministic resolver answers and no Skill affects behavior.

## Example

```bash
uv run --no-sync agentx doctor --probe          # confirm the endpoint accepts both shapes
uv run --no-sync agentx serve
```

Ask a question with and without the Skills through the same run:

```bash
curl -s localhost:9000/api/v1/runs/$RUN/questions \
  -H 'content-type: application/json' \
  -d '{"text":"Where is the item that is currently out of view?","at_ms":6800,"use_skills":true}' \
  | jq '{answer, skills: [.skills[].name], tools: [.tools[].name]}'
```

The response carries `skills[].sha256`; recompute it with
`shasum -a 256 src/agentx/skills/<name>/SKILL.md` to confirm which bytes were in the prompt.

## Measured effect and limitations

The controlled three-arm comparison ran the deterministic resolver, the planner without
Skills and the planner with Skills over identical frozen cases. Across 168 cases per arm, the
Skills reduced rejected selection plans from 36 to 25, and the planner passed 163/168 without
them and 165/168 with them. Both planner arms passed 25/28 on the held-out suite. A later
paired run on the same 28 held-out prompts scored 18/28 for the deterministic resolver and
28/28 in both planner arms, while the Skills arm used more prompt tokens. The Skills therefore
show no held-out correctness gain; the code validators are the backstop. The summary and its
limits are in the [competition results](../../../docs/benchmarks/INDEX.md#agents-and-skills).

Known limitations, all observed rather than assumed:

- A Skill is text in a prompt. It changes what a model tends to produce; it never guarantees
  it. Every property that must hold is also enforced in code, which is why a Skill failure
  degrades into a refusal instead of a wrong answer.
- Instructions embedded in a user's question can still be compiled into selection criteria.
  The held-out suite records two such cases; both ended in a refusal, not a false location.
- The ablation used generated fixtures and one planner model (Nemotron). They do not establish
  real-world accuracy, and they do not separate the contribution of individual Skills.
- The Skills are written for this memory contract. Reusing one elsewhere means re-reading its
  assumptions about cutoffs, registration time and evidence scope.

## Governance files

Each Skill directory also carries the files of the NVIDIA Verified Skills catalog layout. They
sit beside `SKILL.md` and never change it; `SkillRegistry` reads only `SKILL.md`.

| File | Contents |
| --- | --- |
| `skill-card.md` | Skill Card: description, owner, license, use case, credentials, risks and mitigations, output, version and SHA-256. Team-authored, not NVIDIA-verified. |
| `evals/evals.json` | Cases in which the application must load the Skill and negative cases in which it must not, each with the request that triggers it. `tests/test_skill_package.py` executes the routing of every case against the controlled fixture. |
| `BENCHMARK.md` | Only figures already published in the Skills ablation, with the report hash, and "not separately measured" where no measurement isolates the Skill. |
| `skill.oms.sig` | Detached OpenSSF model-signing signature over every file in the directory, verifiable with `skills/findback-skills.pub`. |

The `SKILL.md` files keep their top-level `version` field, which the agentskills.io reference
validator rejects. They are not edited, because saved answers cite the SHA-256 of their exact
bytes. Changing any file in a Skill directory requires signing it again; see
[verifying the signatures](../../../skills/README.md#verify-the-signatures). The portable skill
for coding and chat agents is [`skills/findback-video-memory`](../../../skills/findback-video-memory/SKILL.md).
