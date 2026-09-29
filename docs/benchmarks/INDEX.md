# Competition results and limits

This page contains the conclusions needed to interpret FindBack's competition claims. The measurements use distinct generated fixtures, public benchmark videos and real desk clips. They are not interchangeable. Raw runs, local paths, timestamps, exploratory tools and internal deployment records are kept outside the public source tree.

## Summary

| Area | Headline result | Scope |
| --- | --- | --- |
| Real clips | SAM 2.1 answered **11/11** held-out and **3/3** development questions with no wrong location; classical tracking **9/11** | Five held-out and one development licensed stock clip, frozen coarse labels |
| Live capture and capacity | Warm 5 FPS ingestion p95 **0.818 s / 1.016 s** for one / two targets, **134/134** frames; detector **5.61** samples/s at five targets | One client; present one or two targets at 5 FPS |
| Identity and camera motion | **0** false-visible claims over 40 absent samples; camera recovery **78.9% → 89.1%** recall (held-out cohort 62.6% → 64.4%) | Public LaSOT footage; look-alike recall stays low |
| Agents and Skills | Planner **28/28** vs deterministic **18/28** on indirect descriptions; portable Skill **1/15 → 15/15** per coding agent; internal Skills: no held-out gain | Generated fixtures and server tasks |
| Models and optimization | CUDA mask stage **12.08×** with 831/831 identical predictions; LoRA reviewer **297 → 356/408** held-out but failed its false-presence gate; base-presence fusion is opt-in | Generated masks and LaSOT cohorts |

## Object memory on real clips

Six licensed stock desk clips were used; none of them tuned the tested questions. On the five held-out clips, the SAM 2.1 backend answered **11/11** frozen location questions with no wrong location. The sixth clip is a development clip with three more questions. Classical tracking answered **9/11** and explicitly refused the other two. Replaying those clips as a 5 FPS camera produced **14/14** correct answers while capture was open, and every sealed capture rebuilt to the same observations. The live replay measured one client and at most two registered objects. This demonstrates the scoped workflow on a small set; it does not establish accuracy on arbitrary footage.

The final guarded build was rerun on those same six uploads while the local model services were resident. It answered **11/11** reserved evaluation questions and **3/3** development questions. Median upload, registration and indexing together took **9.697 seconds** (range **5.345–18.162 seconds**). Median deterministic question time was **0.007 seconds**. There were no unanswered questions, false-visible states, wrong zones, wrong last-seen times, or unsupported location claims under the frozen coarse labels. This is a deployment regression on a reused cohort, not a new held-out accuracy result. The labels do not adjudicate whether a visually similar physical object was swapped, and human frame-by-frame task time has not been measured.

## Camera movement and identity

On a paired public video stress test, camera pose recovery improved visible target recall from **78.9% to 89.1%** over **9,349** samples, with 953 gained and none lost across 33 clips. Most of the gain came from development cohorts. On the held-out cohort, recall rose only from 62.6% to 64.4% (18 samples gained), and the confirmation cohort was unchanged. A supported transform maps a detected position back to the registration view; if the relation is unavailable, the region is withheld. It does not prove a metric camera pose.

Identity remains a major failure mode. Similar objects and occlusion can produce a plausible but wrong track. A presence guard and optional same category distractor handling can turn some of these cases into explicit uncertainty, but neither guarantees physical identity. A larger tracker, a simple appearance veto and point tracking did not pass their promotion gates, so they are not claimed as fixes.

On three unmodified LaSOT tracking sequences encoded from ordered source frames at an assigned rate, the guarded SAM 2.1 Small profile made **0 false-visible claims over 40 annotated absent samples**. It localized the target at IoU ≥ 0.5 on **217/229** visible samples in `cup-13`, **117/225** in the harder lookalike `cup-20`, and **218/275** in `bottle-18`. The `cup-20` guard abstained as `identity_ambiguous` on **112** samples after a companion-cup collision; at 30 and 40 seconds, queries returned `unknown` and cited only the earlier 23.6-second confirmed sighting. The low recall in `cup-20` is a real limitation. Review of seven positive low-overlap source frames found partial masks or occlusion, but was not a full physical-identity adjudication.

A separate six-sequence LaSOT stress run found a late-entry failure under the previous guard: in `cup-3`, a second identical cup entered after registration, the target mask briefly spread across both people, and tracking resumed on the other cup. Source-frame review confirmed the swap. The previous guard published **230/300** visible boxes with IoU < 0.1 against the target label. Adding a persistent quarantine for a mask bounding-box area jump above four times the recent accepted median reduced that count to **0/300**, while correct IoU ≥ 0.5 boxes fell from **51/300 to 40/300** because the system abstained after the break. The other five sequences were unchanged. On four additional sequences selected before their paired run, both variants produced **832/964** correct boxes and no IoU < 0.1 visible boxes. A further paired rerun of the previously inspected `cup-20` collision case produced identical predictions in all **231** rows, including **117/225** correct visible boxes and **0/6** false-visible absent samples. The six- and four-sequence sets have no annotated absent samples, and the new rule may also abstain on legitimate rapid scale changes. It is a safety tradeoff, not a general identity solution.

Simply changing to the installed SAM 2.1 Large checkpoint did not repair those difficult cups under the previous guard. On matched `cup-3` samples it produced **238** boxes at IoU < 0.1, versus **230** with Small; on `cup-20` it localized **27/225** visible samples at IoU ≥ 0.5, versus **117/225** with Small. Large did find slightly more correct `cup-3` boxes (58 versus 51). Combined indexing time was **95.4 seconds** versus **65.0 seconds**. This two-clip check is a rejection of that replacement for this profile, not a broad model ranking.

## CUDA mask processing

SAM2 mask bounding boxes and overlap decisions now use exact integer reductions on CUDA, transferring compact results instead of full-resolution masks to the CPU. On GB10, four generated 1080p masks took **21.120 ms** with the previous CPU scan and **1.748 ms** with CUDA reductions (**12.1×** stage speedup). At 4K, the same four-mask workload took **98.347 ms** versus **8.066 ms**. Each measurement used four warmups and 30 timed repetitions per arm with alternating order and CUDA synchronization. The workload contains three overlapping dense rectangular masks and one empty mask; it is a controlled postprocessing benchmark, not recognition accuracy or whole-application latency. Every box and overlap decision matched the reference.

Three paired full-indexing rounds on the previously inspected `cup-20`, `cup-3` and `bottle-18` sequences retained identical predictions for all **831 observations per round**, including boxes, visibility, regions, reasons and both confidence scores. The median combined indexing time changed from **83.848 to 81.529 seconds**, a **2.8%** reduction on this cohort. Both NVIDIA model services remained resident. The repeated observations are not additional independent accuracy samples, and the difficult cups retain the same low recall and identity quarantine. Reproduce the isolated stage with `scripts/eval/benchmark_mask_geometry.py`; use `compare_tracking_reports.py --require-identical-predictions` for the full-pipeline equivalence gate. On the final build, re-indexing the demo clip produced 129 observations in 16.1 seconds, identical row for row to the earlier run.

## Target count and sampling capacity

A generated stationary 1920×1080 scene was processed with 1, 2, 3, 5 and 10 registered targets at 1, 2 and 5 FPS source timestamps. Two rounds reversed trial order, with 40 warmup and 96 measured samples per trial. The real SAM 2.1 small detector and appearance checks ran on GB10, with both NVIDIA model services resident but not receiving requests during measurement.

| Registered targets | Service samples/sec, 1 FPS timestamps | 2 FPS timestamps | 5 FPS timestamps |
| --- | ---: | ---: | ---: |
| 1 | 15.13 | 15.13 | 15.11 |
| 2 | 10.63 | 10.63 | 10.64 |
| 3 | 8.23 | 8.24 | 8.22 |
| 5 | 5.61 | 5.61 | 5.61 |
| 10 | 3.12 | 3.12 | 3.09 |

Values are medians of two unpaced trial throughputs. Changing arrival cadence does not make each inference faster. This is detector capacity, excluding decoding, HTTP and database work; it is not an accuracy benchmark. A separate paced 5 FPS check processed 150 frames each for one and two targets, with **68.6 ms and 96.9 ms p95 completion lag** respectively, excluding warmup and ingestion. All 30 matrix trials retained bounded 34-frame caches.

Keep the live demonstration to **one or two targets at 5 FPS**. Five targets consume about 89% of the detector's frame-time budget before application overhead. Ten targets cannot sustain 5 FPS in this configuration; use 2 FPS for further full-path testing. Applying measured ten-target service times to an initially empty serial queue at 5 FPS produces approximately 12 seconds of final completion lag over 96 arrivals; that backlog is simulated, not a camera measurement. Reproduce these checks with `scripts/eval/benchmark_tracking_capacity.py` and its `--paced` mode. The older 3.24 samples/sec resource soak used 800×480 input and is not a matched resolution comparison.

A separate paced replay sent the licensed phone source through the actual live-ingestion API at 5 FPS and the browser's 1280-pixel capture limit. After prewarming the tracker, one and two targets accepted **134/134 frames each with no capture drops**; p95 frame-to-memory delay was **0.818 s and 1.016 s** respectively. All **9/9** labeled live answers passed, as did the same checks after sealing and rebuilding. Live and rebuilt observations matched exactly for object, timestamp, box, zone, reason and visibility; floating-point scores were not compared. These are single-client development-clip checks with deterministic query resolution. A separate cold-start run reached **5.608 s maximum delay** and dropped two capture frames, so warm the tracker and let the memory watermark catch up before presenting. Detector timing alone does not describe application latency.

## Agents and Skills

The portable FindBack Skill was compared with no Skill on the same prompts and server. Across three held out sets, positive answers improved from **1/15 to 15/15** for each of two coding agents, Claude Code and Codex (exact McNemar p = 0.000122 each). No access token appeared in 200 measured runs. One agent loaded the Skill for one of 15 negative prompts and still refused an unsupported live location claim. A smaller local model benefited less and overactivated, so the result is agent dependent. In an internal three arm study on generated fixtures, Skills reduced rejected planner selection plans from **36 to 25**; final held out correctness remained **25/28** in both planner arms. The validator absorbed many model errors before an answer reached the user.

A later single pass on the same fixed 28-prompt generated holdout with the current Spark planner scored deterministic retrieval **18/28**, planner without Skills **28/28**, and planner with Skills **28/28**. Neither planner arm chose a wrong object or fell back. The Skills arm used **96,060** prompt tokens versus **72,259** without Skills, with 10.1-second versus 7.9-second p95 answer latency. Each planner arm had one invalid selection plan and one unsupported citation rejected by code. This run supports the planner's value on indirect selection, but no incremental correctness gain from the Skills on that set. It is one model run, not a stable latency comparison across hardware conditions.

After adding a bounded "tool box" spelling and name alias, the reused 24-question direct development contract scored **24/24 actual Nemotron answers** both without and with query Skills, with zero fallbacks, wrong selections or missed unique matches in either arm. The deterministic resolver scored **23/24**; its remaining miss is a separate instruction-attack phrasing. The previously failing alias case passed in both planner arms. Median answer time was **3.939 versus 4.690 seconds** and nearest-rank p95 was **6.751 versus 9.012 seconds**, without versus with Skills, in two serial runs on the same server. This repairs an inspected development failure; it is not a new held-out accuracy result or an isolated latency study. The full private reports retain per-question model output, code hashes and settings.

With query Skills on the same fixed application and Nemotron configuration, all five reused planner development suites totaled **118/120 application passes**, **116/120 passing actual-model answers**, three fallbacks, two missed unique matches and **zero wrong-object selections**. The direct, selection, indirect, confirmation and renamed-inventory suites respectively passed 24/24, 24/24, 15/16, 24/24 and 31/32. The selection suite's two fallbacks passed locally; the renamed-inventory fallback missed its expected object. These are inspected generated-fixture regressions, not independent language or visual accuracy. The machine-readable suite summary, full model outputs and failed cases are retained in private evaluation reports.

StepFun 3.7 Flash (`step-3.7-flash`, a hosted API called from the Spark) was evaluated both as an alternative planner and as an outer Skill agent:

- **As planner, first paired run:** on the same 28 generated held-out prompts, it passed 24/28 with five fallbacks, against 25/28 with two for Nemotron.
- **As planner, strict guard:** after a stricter guard against truncated output, it passed 17/28 with 16 fallbacks.
- **As planner, low reasoning effort:** with `reasoning_effort=low` and a 4,096-token cap, it passed **28/28**, and **11/11** on literal and region scope questions, with no fallbacks. Nemotron also scored 28/28 and 11/11.
- **As outer agent:** where FindBack was listed beside three NVIDIA VSS Skills, it chose the right first action on 14/15 and 7/7 routing prompts. In OpenClaw it answered one generated positive question with the original frame and refused a clean negative.

Nemotron remains the production planner. The favorable StepFun profile is a diagnostic on reused, team-authored cases, not a ranking. In the same OpenClaw setup, a Spark-hosted Nemotron outer agent answered its positive case. It timed out on a negative that reused a workspace containing earlier evidence; a narrow host guard now refuses that request class.

The deterministic resolver remains the fallback when the planner is unavailable, produces invalid steps repeatedly, or exceeds its bounded tool budget. A zero-match description gets no location; multiple matches require a user selection. The object-selection Skill is withheld for an explicit object ID or exact registered name; the tool-loop Skill is withheld from deterministic retrieval; the visual-review Skill is withheld from authoritative answers. These negative triggers and the loaded Skill hashes are specified in the [Skill routing contract](../../src/agentx/skills/README.md).

A later live integration check exposed an invented exact-one event limit for "the item that disappeared," which incorrectly excluded an object with two recorded loss events. The planner now receives explicit count guidance, and code rejects positive event-count caps without count language. Re-asking the same saved memory correctly distinguished unrestricted disappearance, exactly once and twice in **3/3 actual Nemotron answers**. The reused selection-confirmation and indirect suites scored **24/24 and 15/16**, with no fallback or wrong-object answer. At that stage, the remaining "never left the center area" miss weakened a history condition and returned ambiguity. The first failed smoke receipt is retained; these are inspected regression results, not independent language accuracy.

Further language checks now preserve explicit region-history conditions, distinguish “the only item” from “only ever seen,” and reject a generic “registered” name filter. Time repair separates opening-frame state from registration at time zero, and preserves exclusive before/after boundaries. An earlier 132-case regression pass produced **131/132 application passes**, **129/132 passing actual-model answers**, three fallbacks and no wrong-object positive selections; its remaining registration-time miss exposed misleading repair feedback. After that correction, the affected temporal, region-history and holdout suites passed **60/60 actual-model checks with no fallback**. Replaying two preserved invalid plans also produced **2/2 successful actual-model repairs** after rejection. Those injected-plan checks are separate from ordinary query accuracy. Two correct fallback cases from the broader run remain disclosed: a malformed direct-history envelope and an unsupported physical claim. These are inspected, generated-fixture regressions; the final 60-case rerun is not a fresh execution of all 132 cases.

## Response time

On an earlier reviewed build, two generated scenes were reference-indexed. **36/36 warm mixed HTTP queries** returned through the live loopback app with no request failures and no planner fallbacks. Median latency was **5.535 seconds**, and the corrected nearest-rank p95 was **12.960 seconds**, below an internal 15-second target. Two answers showed warnings because code had removed unsupported model citations; their saved answers kept valid evidence. The original measurement printed 12.711 seconds because it selected rank 34 instead of rank 35 of 36. The evaluator now uses nearest rank, and the private per-query summary keeps both values. After an app restart, 39 saved questions across the two scenes, both run records and a cited source JPEG in each scene had unchanged digests.

A later intermediate build, run on a separate loopback workspace, returned **36/36** warm mixed questions with no request failures or fallbacks. Median latency was **6.030 seconds** and nearest-rank p95 was **7.930 seconds**. The final build has no separate warm-latency distribution. All of these are single-client, generated-fixture results, not cold-start, concurrent or user-perceived latency. The difference between the two runs is not an isolated speedup, because model state and runtime conditions were not paired.

## Visual reviewer and model tuning

All reviewer candidates were scored on the same frozen reference-crop point-in-box windows from public LaSOT footage. The development cohorts have 366 visible points and 18 absent requests:

- The deployed **Cosmos-Reason2-8B** localized **270/366** points, with **5/18** false "visible" claims.
- NVIDIA **Cosmos3-Nano** localized **263/366**, with **8/18**.
- StepFun **Step3-VL-10B** localized **249/366**, with **8/18**. It was served locally with a no-think chat template, which cut one grounding frame from 110.7 to 3.3 seconds.

A separate held-out cohort has 17 sequences and 408 visible points, with no absent frames. There, Cosmos3-Nano localized **352/408**, against **297/408** for the base and 288/408 for Step3-VL. Cosmos3-Nano made more false "visible" claims on the development absent requests, so it was not promoted.

Two serving changes were measured. FP8 online quantization cut the reviewer weights from 16.74 to 10.04 GiB, but it lost a net 14 development points and raised false "visible" claims from 5/18 to 7/18. The service therefore stays in bf16. Running eight concurrent review workers instead of four reduced the median time per 8-frame window from 6.731 to 4.158 seconds, with no changed answer.

The team fine-tuned Cosmos-Reason2-8B on the GB10 with LoRA:

- **Recipe:** rank 16, vision tower frozen, one epoch, and the training process capped at 55% of unified memory. The samples map a reference crop to a point and were derived from GOT-10k.
- **Promotion gate, frozen before training:** held-out recall must rise by at least five points, precision must not fall, and false claims on development absent frames must not increase.
- **v1 (5,000 samples):** held-out hits rose from **297 to 356/408**, but false "visible" claims rose from **5 to 9/18**.
- **v2 (rebalanced):** **352/408**, with **7/18** false claims.
- Both adapters failed the gate, so neither replaced the base model.

A narrower combination was then tested prospectively on ten unused bottle sequences: the base model decides whether the object is present, and the v1 adapter supplies the point. Correct points rose from **167 to 185/240**, and present claims on different-recording reference proxies fell from **86 to 75/240**. This combination passed its relative gate and is served as an opt-in reference-review mode. Those proxies do not prove physical identity, and many false claims remain.

A third adapter, trained with 900 look-alike hard negatives, sharply reduced proxy claims (guitar 158 → 89, electric fan 138 → 24 of 168). It lost visible localization on every cohort, however, and failed the gate. A base-point plus v3-veto rule proposed afterward also failed its prospective test.

Visual reviews remain advisory and cannot rewrite object memory. In one real free-form review of the demo clip's removal moment, Cosmos described the phone as undisturbed while its own cited frame showed the phone gone. The memory answer, last seen at 13.4 seconds, was unaffected.

## What remains unmeasured

- A blinded study of people using their own recordings, including full registration and indexing time.
- Broad accuracy across camera layouts, lighting, object categories and long term occlusion.
- A matched end to end comparison with NVIDIA VSS on the same original footage and tasks. VSS offers wider video search and Q&A; FindBack's measured focus is registered item history, historical cutoffs and checkable citations.

The [evaluation method](../EVALUATION.md) explains how to reproduce a controlled software check and design a held out real data test. The [architecture](../ARCHITECTURE.md) explains which claims are enforced by code and which remain model judgments.
