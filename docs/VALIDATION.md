# Validation summary

The public acceptance record keeps software contract checks separate from claims about model and user outcomes.

| Area | Evidence | Boundary |
| --- | --- | --- |
| Controlled sample | Backend, browser and API checks cover registration, immutable runs, cutoff-scoped answers, evidence replay, live capture and export verification | Generated fixture; not real-world accuracy |
| Final source checks | `make test`: 632 passed, 14 environment-dependent skips in a clean CPU-only checkout (13 optional PyTorch tests and one optional signing runtime). `make check`: privacy, ruff, mypy, Svelte and formatting checks pass. All eight Skill signatures verify against the published key | The 12 Playwright workflows also run in CI; a CI pass is not a model-quality result |
| Real desk clips | On the final build, SAM 2.1 answered 11/11 held-out and 3/3 development questions with no wrong location. An earlier live replay answered 14/14 while capture was open | Five held-out clips and one development clip, all licensed stock; one client; a small object set |
| Live ingestion | Warm 5 FPS replay through the live API: 134/134 frames with no drops; p95 frame-to-memory 0.818 s for one target and 1.016 s for two; 9/9 labeled answers live, after sealing and after rebuild | One development clip, one client; cold start reached 5.608 s and dropped two frames |
| CUDA mask stage | Every box and overlap decision matched the CPU reference. Three paired full-indexing rounds kept all 831 predictions identical, with 2.8% less indexing time. Re-indexing the demo clip reproduced all 129 observations | Generated masks and three public tracking sequences; not an accuracy gain |
| Target-count capacity | Detector throughput was measured for 1–10 targets at 1/2/5 FPS timestamps; all 30 trials kept a bounded 34-frame cache | Detector only; the live demo scope is one or two targets at 5 FPS |
| Camera movement | A paired stress test improved visible-target recall from 78.9% to 89.1% over 9,349 samples (held-out cohort 62.6% to 64.4%) | Public benchmark footage, mostly development cohorts |
| Agents and language | The planner answered 28/28 indirect descriptions against 18/28 for the deterministic resolver. A 132-case language regression passed 131/132 in the application, with no wrong-object selection. The affected suites then passed 60/60 on the final source | Generated and inspected fixtures; not a fresh 132-case run on the final source |
| Agent Skill | Held-out positive prompts improved from 1/15 to 15/15 for each of two coding agents | Agent dependent; a smaller local model had more failures |
| Deployment | On the Spark, a CUDA SAM 2.1, Nemotron and evidence-report smoke test passed 3/3 scenarios. The deployed runtime and built web files match the release manifest, and existing saved answers were unchanged after deployment | Integration check on a generated fixture |
| Evidence export | Tests verify member hashes, source scope, citation cutoff and offline report integrity; altered media is rejected | Integrity does not prove that a model identified the right object |

The [competition results](benchmarks/INDEX.md) give the denominators and remaining limitations. The [evaluation method](EVALUATION.md) explains how to run the controlled regression and how to design a broader held-out study.
