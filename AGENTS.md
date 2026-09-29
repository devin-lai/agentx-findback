# AgentX contributor guidance

Read `CONTRIBUTING.md`, `docs/README.md` and the relevant architecture documentation before changing behavior. Keep shared documentation, code and UI copy in English.

- Preserve existing uncommitted work. Keep changes focused on the requested task.
- Preserve original-source evidence, query cutoff and recording/run boundaries. Model interpretations stay separate from verified observations.
- Add database changes as new migrations and verify affected API/browser workflows.
- Keep personal plans, raw experiment receipts, exact experiment dates, tool transcripts, submission drafts and machine access notes under ignored `.private/` or `artifacts/`. Public docs contain reviewed competition conclusions and the details needed to understand them.
- Do not read or print credentials. Keep `.env.example` free of real credentials and machine-specific values.
- Run appropriate checks from `CONTRIBUTING.md`, including `make privacy-check` before preparing a commit or release.
- Keep scripts grouped by role; see `scripts/README.md` for their entry points.
