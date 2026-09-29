# Evaluation method

## Keep evidence classes separate

The built-in generated fixture tests software contracts: registration, time cutoffs, movement, disappearance, historical answers and citation replay. It says nothing about general object-recognition accuracy.

Public benchmark videos add externally authored boxes and visibility flags. They support paired tracking, camera movement and reviewer studies on named cohorts. They do not replace original footage or a study of people using the product. The [competition results](benchmarks/INDEX.md) report the measured outcomes and retained limits.

An end to end product study needs authorized original recordings, independent object/time/region labels and participants who did not annotate the clips. Separate development from held out recordings. Freeze labels and scoring rules before the first measured run; count failures and refusals in the denominator. Never treat a model answer as ground truth.

## Controlled software regression

Run the generated fixture with no model weights:

```bash
uv run --no-sync python scripts/eval/evaluate.py --output artifacts/fixture.json
```

The result checks source scope, query cutoff, selected object, status, zone and cited observation. A failure remains in the output and returns a nonzero exit code. `artifacts/` is ignored. Use `make test` for backend contracts, `make e2e` for browser workflows and `make privacy-check` before preparing a source release.

For a paired Skill comparison, answer the same frozen questions against one fixture and memory run with the planner Skill bodies loaded and withheld. Keep the model, tool set and budget fixed. Inspect planner fallbacks separately from correct model actions; code guards continue to enforce source and time rules in both arms. The three-arm study summarized in the competition results did not show a held-out correctness gain, although it reduced invalid selection plans from 36 to 25. A later paired run on the same 28 held-out prompts scored 28/28 in both planner arms and 18/28 for the deterministic resolver.

## Real footage scoring

For each clip, record the licensed source separately from labels. Each question needs a chosen cutoff, expected object, supported status, last seen time tolerance and region when a region is supported. Score at least:

- Correct object and intent.
- Status and last supported region at the question cutoff.
- Error in the last seen timestamp.
- False claims of visibility or a current location after the object leaves view.
- Whether each citation points to the original source at or before the cutoff.
- Full task time, including registration and indexing when comparing user workflows.

Report the number of clips, questions, visible and absent samples, the model or backend profile, and limitations alongside every rate. Avoid combining generated checks, public benchmark frames and original footage into one accuracy percentage. A successful report integrity check proves that saved files match their manifest; it does not prove that the tracker followed the right physical object.

Raw recordings, annotation drafts, row level predictions and exact experiment metadata remain in ignored local storage. Public docs contain the reviewed conclusions needed to interpret the competition claims.
