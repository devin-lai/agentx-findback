# Portable Agent Skills

[`findback-video-memory`](findback-video-memory/SKILL.md) lets a Skills aware agent ask a FindBack server about one registered object in one recording and retrieve the cited original frame. [`findback-evidence-audit`](findback-evidence-audit/SKILL.md) checks an exported evidence ZIP offline without executing code from it. Both are team authored Agent Skills, not NVIDIA verified Skills.

Each package contains `SKILL.md`, a Skill Card, positive and negative evaluation tasks, a benchmark summary and a detached team signature. The video memory Skill also contains a standard library client and short references. The [benchmark summaries](findback-video-memory/BENCHMARK.md) report measured benefits and failures without raw transcripts. In the video memory benchmark, the smaller local model's 2/16 → 5/16 row counts 4 held-out and 12 development positives. On a later held-out set, after the read-and-ask token was introduced, that model went from 0/5 to 3/5 with no server writes. The signed benchmark file is unchanged.

## Use the server Skill

1. Start FindBack and set `FINDBACK_URL` to the trusted server. Set `FINDBACK_TOKEN` only if that server requires authentication; prefer its read and ask token for agent access.
2. Install or expose the `findback-video-memory` directory to a Skills aware agent.
3. Ask about an object registered in a selected recording and provide a query cutoff when needed. Inspect the returned frame before relying on a physical identity claim.

The Skill cannot establish where an object is now from old footage. It does not search across an archive, identify people, or infer an unseen destination. The server enforces citation scope independently of the agent.

### Reproduce the generated workflow

From the repository root, follow the local setup in the main README and start
`agentx serve` with an empty `AGENTX_DATA_DIR`. Set the same absolute
`AGENTX_DATA_DIR` in a second terminal, set `FINDBACK_URL` to that server, and
run:

```bash
python3 skills/findback-video-memory/scripts/findback.py demo --scene fixed
python3 skills/findback-video-memory/scripts/findback.py index <video_id> --backend reference --reuse --wait
python3 skills/findback-video-memory/scripts/findback.py ask --video "Controlled desk fixture" "Where was Red toolkit last seen?" --at-ms 7000 --no-planner --save-frame artifacts/skill-example-output
python3 skills/findback-video-memory/scripts/findback.py report <question_id> --out artifacts/skill-example-output/evidence.zip
python3 skills/findback-evidence-audit/scripts/audit_report.py artifacts/skill-example-output/evidence.zip --source "$AGENTX_DATA_DIR/videos/<video_id>/source.mp4"
```

Use the IDs printed by `demo` and `ask`. The [actual generated-fixture
trace](examples/controlled-desk-trace.json) records a last-supported sighting
at 5.8 seconds for a 7-second question, the application Skill/tool path, the
source and report SHA-256 values, and a verified source-video audit. Its
SHA-256 is `2a09247d93325c5069a8e0532f9c360ab5a707231edf27008720b9b7d95737f7`.
New runs create new IDs and report hashes. This example uses local reference
tracking and no planner; it demonstrates the portable commands and evidence
contract, not Nemotron quality or real-footage accuracy.

## Verify an exported report

Install or expose `findback-evidence-audit`, then give the agent the ZIP path, and optionally an original video and trusted ZIP hash. The auditor checks internal integrity and cutoff scope. Its [benchmark summary](findback-evidence-audit/BENCHMARK.md) explains the limits of those checks.

## Verify the signatures

Each Skill directory, both here and under `src/agentx/skills/`, has a detached `skill.oms.sig`. It is an OpenSSF model-signing signature over every file in that directory, made with the team's published key [`findback-skills.pub`](findback-skills.pub). To verify all eight packages, use an interpreter that has `model-signing` installed. It is deliberately not a project dependency:

```bash
python3 -m venv /tmp/sigvenv && /tmp/sigvenv/bin/pip install "model-signing==1.1.1"
/tmp/sigvenv/bin/python scripts/ops/sign_skills.py verify
```

Verification is strict: a changed or added file fails. Remove stray `.DS_Store` or `__pycache__` entries before verifying a copied directory. A signature proves package integrity relative to that key; it is not third-party verification or a safety review. See [repository privacy](../docs/REPOSITORY_PRIVACY.md) before publishing new evaluation material.
