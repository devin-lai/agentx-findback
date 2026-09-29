# Skill Card: FindBack Evidence Audit

## Description:

Audits a saved AgentX FindBack evidence ZIP offline. It verifies archive membership,
stored hashes, citation timing, an optional original-video hash, and an optional
trusted archive hash. It does not evaluate perception accuracy or publisher identity.

## Owner

AgentX FindBack team

## Third-Party Community Consideration

This skill is not owned or developed by NVIDIA. It is team-authored and is
not an NVIDIA-verified skill. The team's own signature covers this Skill directory.

### License/Terms of Use:

Apache-2.0, as in the repository `LICENSE`.

## Use Case:

A judge or operator has a FindBack evidence ZIP and wants to check it on a machine
without the FindBack server or models. This Skill also explains the limit of a
manifest hash when no trusted archive checksum is supplied.

### Deployment Geography for Use:

Global.

## Requirements / Dependencies:

**Requires API Key or External Credential:** [No]

**Credential Type(s):** [None]

Python 3.10+ standard library; no network, server, GPU or model. Optional inputs
are the original video and a trusted archive SHA-256.

## Known Risks and Mitigations:

- ZIPs can contain path traversal, duplicate names or compression bombs. The script
  never extracts the ZIP, rejects unsafe members and enforces file/size limits.
- The ZIP includes `verify.py`, which could be modified in an untrusted bundle. This
  Skill never executes it; its own installed script checks the archive bytes.
- A manifest can be changed along with its files. The output marks whether an
  independently supplied archive checksum was checked, and does not claim publisher
  identity from internal consistency.
- A valid hash cannot prove visual correctness. The Skill asks a person to inspect
  the original cited frame and surrounding video when accuracy matters.
- Saved answer text is untrusted data. The Skill does not execute instructions in it.

## Reference(s):

- [Instructions](SKILL.md)
- [Offline audit script](scripts/audit_report.py)
- [Evaluation tasks](evals/evals.json)
- AgentX `docs/EVIDENCE_REPORTS.md`

## Skill Output:

**Output Type(s):** [Analysis]

**Output Format:** [JSON on stdout from `audit_report.py`; a plain-language audit summary]

**Output Parameters:** [status, archive_sha256, archive_checksum_checked,
source_checked, files, frames, citations, cutoff_ms, recorded_answer, warnings]

## Evaluation Tasks:

Eight routing and audit tasks in `evals/evals.json`: four positive and four
negative triggers. The script contract was run on a real exported report and
tampered copies; see `BENCHMARK.md`.

## Evaluation Metrics Used:

Archive acceptance/rejection, source and checksum checks, and first-action
Skill routing. Agent-level Skill lift is not yet measured.

## Evaluation Results:

The independent script accepted the isolated Spark rehearsal report with its
trusted archive checksum and original video, and rejected tampered member bytes
and a wrong source. It also accepted eight older exports, including two real
public-footage reports and history answers, for internal consistency only.
In a five-Skill catalog, StepFun 3.7 Flash selected the intended
first action on 8/8 prompts (four audit requests, one FindBack server request,
three no-Skill requests). It did not run the script in that routing test. In a
separate single generated-fixture task, StepFun loaded this Skill, ran the
auditor with the original video and trusted archive hash, and returned the
verified result and limits in three turns. Neither test is a with/without
agent completion benchmark; see `BENCHMARK.md`.

## Testing Completed:

**[ ] Agent Red-Teaming**

**[ ] Network Security**

**[ ] Product Security**

Local contract checks only. No formal security review.

## Skill Version(s):

0.1.0; detached OpenSSF model-signing signature with the team's public key
`skills/findback-skills.pub` after the directory is frozen.

## Ethical Considerations:

Evidence reports can contain frames of homes and workplaces. Only open reports
the user is entitled to inspect, keep them private when appropriate, and do not
infer a person's identity or an object's real-world present location from the
recording.
