.PHONY: setup setup-vision web serve check privacy-check test evaluate e2e format release release-verify node-setup node-models node-services node-start node-stop node-smoke

setup:
	uv sync --locked --python 3.12
	npm --prefix web ci

setup-vision:
	uv sync --locked --extra vision
	npm --prefix web ci

web:
	npm --prefix web run build

serve:
	uv run --no-sync agentx serve

privacy-check:
	python3 scripts/ops/check_repository.py

check: privacy-check
	uv run --no-sync ruff check src tests scripts skills integrations
	uv run --no-sync mypy src/agentx
	npm --prefix web run check
	npm --prefix web run format:check

test:
	uv run --no-sync pytest -q

evaluate:
	uv run --no-sync python scripts/eval/evaluate.py

e2e: web
	cd web && npm run test:e2e

format:
	uv run --no-sync ruff format src tests scripts skills integrations
	npm --prefix web run format

release: web
	uv run --no-sync python scripts/ops/package_release.py
	uv run --no-sync python scripts/ops/verify_release.py artifacts/releases/agentx-findback.tar.gz --root .

release-verify:
	uv run --no-sync python scripts/ops/verify_release.py artifacts/releases/agentx-findback.tar.gz --root .

# DGX Spark node (run on the node from the synced checkout; user-level only)
node-setup:
	bash scripts/spark/setup_node.sh

node-models:
	bash scripts/spark/download_models.sh 8B

node-services:
	bash scripts/spark/start_model_services.sh

node-start:
	bash scripts/ops/node_service.sh start

node-stop:
	bash scripts/ops/node_service.sh stop

node-smoke:
	HF_HUB_OFFLINE=1 .venv/bin/python scripts/ops/preflight.py --require-cuda
	HF_HUB_OFFLINE=1 .venv/bin/python scripts/smoke/smoke_vision.py --require-cuda --identity
	.venv/bin/python scripts/smoke/smoke_cosmos.py --grounded --output artifacts/spark/cosmos-8B-grounded.json
