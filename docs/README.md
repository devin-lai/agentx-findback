# Documentation map

| Page | Purpose |
| --- | --- |
| [Project overview](PROJECT_OVERVIEW.md) | Product workflow and measured boundaries |
| [Chinese project description](PROJECT_DESCRIPTION.zh-CN.md) | 中文项目说明: highlights, architecture, deployment, optimization and tech stack |
| [Architecture](ARCHITECTURE.md) | Memory, agent and evidence contracts |
| [Deployment](DEPLOYMENT.md) | Local and DGX Spark setup |
| [Evaluation](EVALUATION.md) | Controlled and held out test methods |
| [Competition results](benchmarks/INDEX.md) | Key outcomes, denominators and limitations |
| [Evidence reports](EVIDENCE_REPORTS.md) | Export and offline verification |
| [Validation](VALIDATION.md) | Public acceptance summary |
| [Repository privacy](REPOSITORY_PRIVACY.md) | Publication boundary |
| [Script guide](../scripts/README.md) | Engineering tool layout |

The application lives under `src/agentx/`, the Svelte client under `web/`, backend tests under `tests/`, portable Skills under `skills/`, and integrations under `integrations/`. `scripts/` groups operations, Spark model serving, data preparation, evaluation, training and smoke checks by purpose. Architecture decisions are in `docs/adr/`.

Public documentation records product behavior and competition conclusions. Raw experiment output, private work logs, exact execution schedules, local machine details and draft submissions belong in `.private/` or `artifacts/`.
