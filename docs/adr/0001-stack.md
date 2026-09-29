# ADR 0001: A modular monolith with replaceable perception

Status: Accepted.

AgentX FindBack starts with uploaded, fixed-camera video. The team has 2–3 developers and access to a DGX Spark. Correct temporal state, evidence and reproducibility matter more than framework count.

## Decision

- Python 3.12, uv and a committed lock file.
- FastAPI and Pydantic for a versioned HTTP API and validated contracts.
- A framework-independent domain layer for temporal memory. SQLAlchemy 2 repositories with Alembic migrations; SQLite WAL initially, a PostgreSQL-compatible schema for later deployment.
- PyAV for presentation timestamps, FFmpeg for browser playback, OpenCV for a dependency-light reference-matching baseline. RT-DETR and ByteTrack are an optional perception adapter with separately installed GPU dependencies.
- Svelte 5, TypeScript and Vite for a small component-based client. No full-stack JavaScript server, global state framework or generated UI runtime.
- An explicit bounded Python workflow loads versioned Skills, plans a query, retrieves typed observations and validates evidence. Planner models are optional OpenAI-compatible adapters; ADR 0002 selects Nemotron on the node. Cosmos visual review supports a lazy native Transformers adapter or an independent HTTP inference service. An unconfigured provider is visibly unavailable; local fallback never claims a model ran.
- Database-backed single-worker jobs for this deployment. An interrupted job is marked failed and can be restarted as a new immutable run. We do not pretend an in-process worker is a distributed queue.
- REST/OpenAPI is the public boundary. Robotics, MCP, streaming and additional model adapters can use the same application services later.

## Trade-offs

SQLAlchemy and migrations cost more setup than raw SQLite but give explicit evolution and transactions. Svelte is a genuine client framework without the deployment surface of an SSR application. The small workflow does not need LangGraph today; any future workflow engine must consume the same services rather than own the memory format. No design eliminates future technical debt: boundaries, tests, versioned records and documented limitations make changes manageable.

Do not add a vector store until semantic candidate retrieval requires one. Similarity is not temporal truth. Do not introduce microservices until process isolation, throughput or independent ownership warrants them.

## Primary references

- https://fastapi.tiangolo.com/tutorial/bigger-applications/
- https://docs.sqlalchemy.org/en/20/orm/quickstart.html
- https://alembic.sqlalchemy.org/en/latest/
- https://svelte.dev/docs/svelte/overview
- https://docs.astral.sh/uv/guides/projects/
- https://github.com/nvidia-cosmos/cosmos-reason2
- https://huggingface.co/PekingU/rtdetr_r18vd
- https://github.com/roboflow/trackers

## Model lifecycle note

The upstream Cosmos Reason2 repository now directs new development toward Cosmos 3. Keep Reason2 compatibility for existing competition assets, but avoid importing its runtime into the memory core. The separate HTTP adapter allows an independently validated model migration. RT-DETR R18 is selected as a small pinned perception baseline, not as a claim to the newest detector.
