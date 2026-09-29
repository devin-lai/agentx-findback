import hmac
import logging
import re
from contextlib import asynccontextmanager
from types import SimpleNamespace

import av
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agentx.agents.agent import ToolAgent
from agentx.agents.providers import Providers
from agentx.agents.skills import SkillRegistry
from agentx.agents.workflow import QueryWorkflow
from agentx.api.routes import router
from agentx.config import Settings
from agentx.services.catalog import Catalog
from agentx.services.discovery import Discovery
from agentx.services.evidence_bundle import EvidenceBundles
from agentx.services.indexing import Indexer
from agentx.services.live import LiveRecorder
from agentx.services.memory import Memory
from agentx.services.reviews import Reviews
from agentx.storage.database import Database

logger = logging.getLogger(__name__)

# The only write an agent token may make: asking a question, which saves an answer and never
# changes what a recording remembers.
AGENT_WRITE = re.compile(r"/api/v1/runs/[^/]+/questions")


def agent_may(method: str, path: str) -> bool:
    return method in {"GET", "HEAD", "OPTIONS"} or (
        method == "POST" and AGENT_WRITE.fullmatch(path) is not None
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.prepare()
    db = Database(settings.db_url)
    catalog = Catalog(db, settings)
    skills = SkillRegistry()
    providers = Providers(settings)
    memory = Memory(db, catalog)
    indexer = Indexer(db, settings, catalog, skills, providers)
    reviews = Reviews(db, catalog, memory, providers, skills)
    agent = ToolAgent(memory, providers, reviews, settings)
    discovery = Discovery(catalog, providers, settings)
    live = LiveRecorder(db, settings, catalog)
    workflow = QueryWorkflow(db, memory, skills, settings, agent)

    @asynccontextmanager
    async def lifespan(app):
        db.migrate()
        # Always run: it also seals captures a previous process left open, even when live capture
        # has since been disabled and its endpoints return 404.
        live.start()
        if settings.worker_enabled:
            indexer.start()
        yield
        live.close()
        indexer.close()
        db.engine.dispose()

    app = FastAPI(
        title="AgentX FindBack",
        version="0.1.0",
        lifespan=lifespan,
        description="Causal, evidence-backed object memory for uploaded video.",
    )
    app.state.services = SimpleNamespace(
        settings=settings,
        db=db,
        catalog=catalog,
        skills=skills,
        providers=providers,
        memory=memory,
        indexer=indexer,
        workflow=workflow,
        reviews=reviews,
        agent=agent,
        discovery=discovery,
        live=live,
        bundles=EvidenceBundles(db, catalog, memory),
    )

    @app.middleware("http")
    async def access_control(request: Request, call_next):
        protected = request.url.path.startswith(("/api/v1", "/docs", "/openapi.json", "/redoc"))
        authorization = request.headers.get("authorization", "")
        has_bearer = authorization.lower().startswith("bearer ")
        if protected:
            if settings.api_token:
                supplied = (
                    authorization[7:]
                    if has_bearer
                    else ""
                    if authorization
                    else request.cookies.get("agentx_session", "")
                ).encode()
                agent = bool(
                    settings.agent_token
                    and has_bearer
                    and hmac.compare_digest(supplied, settings.agent_token.encode())
                )
                if not agent and not hmac.compare_digest(supplied, settings.api_token.encode()):
                    return JSONResponse(
                        {"detail": "Sign in with the configured access token."}, status_code=401
                    )
                if agent and not agent_may(request.method, request.url.path):
                    return JSONResponse(
                        {
                            "detail": "The agent token can read memory and ask questions. "
                            "Uploads, registrations, indexing and other changes need the "
                            "operator's token."
                        },
                        status_code=403,
                    )
            elif (
                not request.client
                or request.client.host not in {"127.0.0.1", "::1", "testclient"}
                or request.url.hostname not in {"localhost", "127.0.0.1", "::1", "testserver"}
            ):
                return JSONResponse(
                    {"detail": "Configure AGENTX_API_TOKEN before network access."}, status_code=503
                )
        # A loopback service is still reachable by a browser visiting another site. Only an
        # authenticated Bearer request may bypass the browser origin check, never a cookie
        # accompanied by an unrelated Authorization header or a token-free local request.
        authenticated_bearer = protected and bool(settings.api_token) and has_bearer
        if (
            request.url.path.startswith("/api/")
            and request.method not in {"GET", "HEAD", "OPTIONS"}
            and not authenticated_bearer
        ):
            origin = request.headers.get("origin")
            if (
                origin and origin.rstrip("/") != str(request.base_url).rstrip("/")
            ) or request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse(
                    {"detail": "Cross-origin writes are not allowed."}, status_code=403
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.exception_handler(av.error.FFmpegError)
    async def undecodable_media(request, exc):
        # Decoder messages name absolute storage paths; keep them in the log, not the response.
        logger.warning("Media decoding failed for %s: %s", request.url.path, exc)
        return JSONResponse(
            {"detail": "The stored recording could not be decoded at this point."},
            status_code=422,
        )

    @app.exception_handler(ValueError)
    async def invalid_request(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(LookupError)
    async def not_found(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    app.include_router(router)
    dist = settings.web_dist.resolve()
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def home():
        if (dist / "index.html").is_file():
            return FileResponse(dist / "index.html")
        return JSONResponse(
            {
                "message": "Build the web client with npm ci && npm run build in web/, then restart the server.",
                "api": "/docs",
            }
        )

    return app
