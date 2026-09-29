# Portable evidence reports

Select **Save evidence report** below any answer in a completed memory version. Extract the downloaded ZIP and open `report.html`. The report works offline in a browser, without AgentX, a model service, or installed packages. The report contains the full scene around the object in each included frame.

Each report packages one answer at its original cutoff:

- The question, supported location, uncertainty, and the memory version's region names.
- Every cited observation, shown chronologically with a removable bounding-box overlay. The JPEG files themselves have no overlays.
- For an uncertain location with a last sighting, a separate source frame at or just before the cutoff, labeled **context only**. This is available for visual inspection, not a new positive observation or a claim of absence.
- The saved planner label, model, tool trace, warnings, and versioned Skill hashes. Planner draft text remains separate from the supported answer.
- Perception model revisions, runtime and measurement settings; host paths and inference endpoints are excluded through a metadata allowlist.
- A machine-readable `report.json`, SHA-256 `manifest.json`, and standalone Python verifier.

No later frames, complete source video, separate Cosmos reviews, credentials, or model weights are bundled. The generated demonstration is explicitly labeled a software fixture. The existing whole-run JSON export remains available for raw observation analysis.

## Evidence contract

`GET /api/v1/questions/{question_id}/bundle` uses the same workspace authentication as the other evidence endpoints. The saved answer's record ID, run, question, and cutoff must agree with storage. For a selected object, the answer text, states, events, and references must exactly match the deterministic renderer over that memory version. Legacy answers that cannot pass this check must be asked again; the exporter does not silently rewrite them.

Live evidence reads also verify the original recording against its ingestion fingerprint. The result is cached in a bounded per-process cache and invalidated by a changed inode, size, modification time or change time. A missing or altered source makes newly queried locations unconfirmed and prevents original-frame retrieval, registration, review and new indexing. Historical saved records remain unchanged. This detects ordinary file damage and replacement; it is not protection against an administrator controlling both data and filesystem metadata.

The exporter independently hashes the original video and compares it with its ingestion fingerprint. Every cited frame must decode at its exact stored presentation timestamp, at or before the cutoff. Missing, altered, mismatched, or unsupported evidence aborts the export. The context image may be the nearest earlier frame; its actual timestamp is recorded. Export does not run a model or mutate the saved answer.

The archive has at most 120 distinct frames and 64 MiB of payload. An oversized request receives HTTP 413 and can be shortened by asking at an earlier cutoff. Only one export is built at a time per application process; a concurrent request receives HTTP 429. Temporary files are removed after delivery or on failure. Responses use `Cache-Control: no-store`.

The report page uses escaped static HTML, local image files, and a restrictive content policy. It loads no scripts, remote fonts, remote images, or analytics. It supports narrow screens, standard browser printing, expandable provenance, and a checkbox to hide object bounds.

## Verify a downloaded report

From the extracted directory, with Python 3.10 or newer:

```bash
python3 verify.py
python3 verify.py --source /path/to/original.mp4
```

The first command checks the complete file inventory, sizes, hashes, frame fingerprints and cutoff. The optional source check compares the original video with the stored source hash. No third-party Python package is required. The result is file integrity relative to the supplied manifest, not a digital signature, proof of authorship, or a perception accuracy score. Keep the ZIP's checksum through a trusted channel when passing a report between people.

Older exports may require Python 3.11 or newer. Re-export a saved answer to obtain the current verifier; do not modify checksummed files in place.

## Controlled integration check

Run `scripts/smoke/smoke_evidence_bundle.py` against a configured local or Spark application. It creates a labeled generated fixture, builds memory, asks scoped questions through the configured agent, downloads the reports and verifies their files and citation timing. Browser and backend tests cover offline rendering, altered sources, forged answers, cross-run or cross-object references, future timestamps, authentication and archive limits.

A node integration run produced three valid reports from a generated fixture with no planner fallback. This establishes the export and verification path for those cases; it is not a real-footage accuracy or latency benchmark.
