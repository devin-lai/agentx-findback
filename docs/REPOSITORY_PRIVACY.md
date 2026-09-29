# Repository privacy and publication

## Public boundary

Public source contains the application, small synthetic tests, portable Skills, integration code, setup guidance and reviewed competition conclusions. The only public benchmark page is [competition results](benchmarks/INDEX.md). It gives denominators, distinctions between data types and material limitations without publishing the execution diary.

| Material | Location | Public |
| --- | --- | --- |
| Product code, tests and dependency locks | `src/`, `web/`, `tests/`, `integrations/`, `skills/` | After review |
| Setup and engineering guides | `README.md`, `docs/`, `scripts/README.md` | After review |
| Blank configuration template | `.env.example` | Yes |
| Raw run reports, row level predictions, protocols, draft text and work logs | `.private/` or `artifacts/` | No |
| Research probes and draft demo film | `.private/engineering/`, `web/demo/` | No |
| Credentials, access notes and machine configuration | `.env`, `.private/` | No |
| Recordings, databases and model weights | `.agentx/`, `data/`, `model/`, `models/`, `artifacts/` | No |

Exact experiment dates, local paths, internal tool transcripts, backup inventories and draft submission material are private. Publish a technical claim only with enough scope, denominator and limitations to interpret it. Public source may name the deployed NVIDIA and perception components because they are necessary to explain and run the competition system.

The legacy locations `docs/ROADMAP.md`, `docs/COMPETITION.md`, `docs/CHANGELOG.md`, `docs/DEVELOPMENT_RUN*.md`, `docs/submission/`, and all detailed `docs/benchmarks/` paths except `INDEX.md` are ignored. Keep new private material under `.private/` so it does not mix with public docs. Generated chart data and raw reports also stay private.

`.gitignore` controls normal adds. `.dockerignore`, the Python source build configuration and the release packager limit what goes into distributed artifacts. `make privacy-check` checks tracked files and untracked publication candidates against the private-path policy. It also flags personal home paths, recognizable credential markers and nonempty credentials in `.env.example`. It reports only file names, line numbers and rule names; matching values are never printed. Private paths, ignored untracked files and symlink targets are not opened.

This check cannot establish that all prose, screenshots, binaries or credentials are safe, and it does not inspect Git history. Review the staged diff, image content and metadata, and the release bundle before sharing. Do not force-add ignored material. Use the public author names `devin-lai` and `wingsuomo` for project attribution; review commit author and committer metadata separately before publishing.

The release packager applies the same content rules to the files it includes, including the built frontend, before writing an archive. This check also works in a source distribution without Git.

## Existing history

Ignore rules do not remove previously committed material from Git history or old archives. Before publishing this repository, review commit history, branches and tags. Use a fresh, reviewed source-only repository or a coordinated history cleanup if any prior revision contains private content. Git author metadata is public as well. Never publish an old branch or archive that still contains private material. Rotate any credential that was previously exposed.
