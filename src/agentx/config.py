from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTX_", env_file=".env", extra="ignore")

    data_dir: Path = Path(".agentx")
    database_url: str = ""
    api_token: str = ""
    # Least privilege for Skills and MCP agents: with this Bearer token a client may read
    # recordings, runs, frames and reports and ask questions, and nothing else. It only takes
    # effect alongside `api_token`, which remains the operator's full-access token.
    agent_token: str = ""
    web_dist: Path = Path("web/dist")
    max_upload_mb: int = Field(default=256, ge=1, le=4096)
    max_video_seconds: int = Field(default=1800, ge=1)
    worker_enabled: bool = True
    # Live camera captures: frames are appended to a growing recording that memory follows.
    live_enabled: bool = True
    live_max_fps: float = Field(default=10, gt=0, le=30)
    live_max_edge: int = Field(default=1280, ge=320, le=4096)
    live_max_frame_bytes: int = Field(default=4 * 1024 * 1024, ge=16 * 1024)
    live_idle_seal_seconds: int = Field(default=120, ge=5)
    # Generic OpenAI-compatible planner used by the tool agent (for example a local vLLM
    # Nemotron service on DGX Spark). The legacy StepFun variables remain accepted aliases.
    planner_base_url: str = ""
    planner_model: str = ""
    planner_api_key: str = ""
    planner_max_steps: int = Field(default=8, ge=2, le=12)
    planner_max_tokens: int = Field(default=1200, ge=128, le=4096)
    # Optional vLLM/Nemotron extension; None preserves generic provider compatibility.
    planner_thinking: bool | None = None
    planner_structured_outputs: bool = False
    # Requires a serving backend with reasoning-parser and thinking-budget support.
    planner_selection_thinking_budget: int | None = Field(default=None, ge=0, le=8192)
    stepfun_base_url: str = "https://api.stepfun.com/v1"
    stepfun_model: str = ""
    stepfun_api_key: str = ""
    stepfun_reasoning_effort: Literal["low", "medium", "high"] | None = None
    cosmos_base_url: str = ""
    cosmos_model: str = "nvidia/Cosmos-Reason2-2B"
    # Optional specialized LoRA for registered-object reviews. The base cosmos_model checks
    # presence first; only when both models say visible is the adapter's point used. Freeform
    # reviews, discovery and memory indexing continue to use the base model.
    cosmos_reference_adapter_model: str = ""
    # Which model's point a fused reference review reports once both say "present": the adapter's
    # (deployed rule) or the base model's, using the adapter only as an identity veto.
    cosmos_reference_point_source: Literal["adapter", "base"] = "adapter"
    cosmos_api_key: str = ""
    cosmos_backend: Literal["http", "transformers"] = "http"
    cosmos_model_path: str = ""
    cosmos_device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    cosmos_max_edge: int = Field(default=448, ge=224, le=768)
    # Applies only to grounded reviews that have a registration reference crop.
    cosmos_reference_grounding: Literal["normalized", "integer_1000"] = "normalized"
    cosmos_reference_max_edge: int | None = Field(default=None, ge=224, le=768)
    # Focus pass: a second, zoomed localization around the coarse point. A small desk object
    # covers very few tokens in a frame downscaled to cosmos_max_edge; re-asking inside a crop
    # taken from the ORIGINAL frame gives the same model a usable number of pixels.
    cosmos_focus: Literal["off", "refine", "confirm"] = "off"
    cosmos_focus_window: float = Field(default=0.34, gt=0.05, le=1.0)
    # Bounded HTTP parallelism within one grounded review; native inference stays serial.
    cosmos_review_workers: int = Field(default=4, ge=1, le=8)
    cosmos_max_new_tokens: int = Field(default=512, ge=128, le=4096)
    # Ask Cosmos to reason inside <think> tags before the JSON object. Costs latency.
    cosmos_thinking: bool = False
    # Upper sampling rate for memory built by Cosmos itself (one request per object and frame).
    cosmos_index_fps: float = Field(default=1.0, gt=0, le=5)
    detector_model: str = "PekingU/rtdetr_r18vd"
    detector_revision: str = "ac77a11ff0170a41b771c03264987f8ce2b0d753"
    # Optional appearance-embedding identity check for learned detections.
    identity_model: str = "facebook/dinov2-small"
    identity_revision: str = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    identity_model_path: str = ""
    identity_enabled: bool = False
    identity_threshold: float = Field(default=0.6, ge=0.1, le=0.99)
    identity_margin: float = Field(default=0.05, ge=0, le=0.5)
    sam2_model: str = "facebook/sam2.1-hiera-small"
    sam2_revision: str = "ee5bba1d82bb8749febdf90f45e84b687142ba03"
    sam2_model_path: str = ""
    sam2_memory_frames: int = Field(default=32, ge=32, le=256)
    # Where the tracking session keeps its memory bank and frames. "auto" keeps them with the
    # model; on a unified-memory GB10 that avoids per-frame copies with identical outputs.
    sam2_state_device: Literal["auto", "cpu"] = "auto"
    sam2_presence_threshold: float = Field(default=0.5, ge=0.1, le=0.99)
    sam2_confident_presence: float = Field(default=0.95, ge=0.5, le=0.999)
    sam2_appearance_threshold: float = Field(default=0.7, ge=0.1, le=0.99)
    sam2_distractor_guard: bool = False
    sam2_max_distractors: int = Field(default=4, ge=1, le=8)
    model_device: str = "auto"
    provider_timeout_seconds: float = Field(default=30, gt=0, le=600)

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{self.data_dir.resolve() / 'memory.sqlite3'}"

    @property
    def planner_endpoint(self) -> tuple[str, str, str]:
        """(base_url, api_key, model) for the tool agent; explicit planner values win."""
        if self.planner_model and self.planner_base_url:
            return self.planner_base_url, self.planner_api_key, self.planner_model
        if self.stepfun_model and self.stepfun_api_key:
            return self.stepfun_base_url, self.stepfun_api_key, self.stepfun_model
        return "", "", ""

    @property
    def planner_available(self) -> bool:
        return bool(self.planner_endpoint[2])

    @property
    def cosmos_available(self) -> bool:
        if self.cosmos_backend == "transformers":
            return (
                bool(self.cosmos_model_path)
                and (Path(self.cosmos_model_path) / "config.json").is_file()
            )
        return bool(self.cosmos_base_url)

    @property
    def sam2_available(self) -> bool:
        return bool(self.sam2_model_path) and all(
            (Path(self.sam2_model_path) / name).is_file()
            for name in ("config.json", "model.safetensors")
        )

    def prepare(self) -> None:
        self.data_dir = self.data_dir.resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "videos").mkdir(exist_ok=True)
