# Contributing to AgentX

AgentX FindBack connects object-memory answers to timestamped visual evidence. Start with the [quickstart](README.md#quick-start-no-gpu-no-model-weights), [repository map](docs/README.md), and [architecture](docs/ARCHITECTURE.md).

## Set up a development environment

Use Python 3.12, uv, Node.js 22.12 or newer, and FFmpeg 6 or newer. From the repository root:

```bash
make setup
cp .env.example .env
make web
uv run --no-sync agentx doctor
```

Run `uv run --no-sync agentx serve` and `npm --prefix web run dev` in separate terminals. The default OpenCV backend and controlled sample work without API keys or downloaded model weights. Optional GPU/model setup is described in [deployment](docs/DEPLOYMENT.md).

## Make a change

Follow the [code of conduct](CODE_OF_CONDUCT.md). Contributions you intentionally submit for inclusion are provided under the project's Apache-2.0 license. Keep upstream copyright and license notices with any third-party code you contribute.

Discuss substantial behavior or architecture changes in a GitHub issue before implementation. Choose a focused branch, keep changes reviewable, and explain the problem, resulting behavior and validation in the pull request. Public issues and pull requests should contain only information intended for collaborators.

Keep code, UI copy and shared documentation in English. Put backend logic in its existing `src/agentx/` module, browser code in `web/src/`, backend tests in `tests/`, and browser workflows in `web/tests/`. Store schema changes as new Alembic migrations. Record substantial architecture decisions in `docs/adr/`.

Preserve source timestamps, recording/run scope, explicit uncertainty and citation validation. A model's response must not replace verified memory or establish an unseen object's current location. See the architecture document for the contracts.

## Verify a change

```bash
make privacy-check
make check
make test
make evaluate
cd web && npx playwright install chromium && cd ..
make e2e
```

Run the checks relevant to your change before opening a pull request; CI runs the complete baseline suite. Use `make format` for Python and frontend formatting. GPU smoke checks and actual-provider benchmarks require separately configured models and are not prerequisites for ordinary contributions. Report which checks ran and any unverified behavior, without exposing personal setup or internal experiment tooling.

## Keep personal material local

Use `.private/` for personal development plans, journals, submission drafts and machine access notes. Use `.env` for local configuration, `artifacts/` for generated outputs, and `.agentx/` for application data. These paths are ignored. See [repository privacy](docs/REPOSITORY_PRIVACY.md) for all publishing boundaries and the limitations of ignore rules.

The only committed environment file should be `.env.example`, with blank credentials and portable examples. Review `git diff --cached` before committing. Do not paste tokens, private endpoints, personal recordings or private plans into issues, screenshots or CI output. If you find a vulnerability, follow [SECURITY.md](SECURITY.md).
