"""Optional, lazy local Cosmos inference. No model download or remote code execution."""

import base64
import hashlib
import io
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image

from agentx.config import Settings


class LocalCosmos:
    """One resident model per process; its own lock serializes loading and generation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: Any = None
        self.processor: Any = None
        self.provenance: dict = {}
        self.lock = Lock()

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        path = Path(self.settings.cosmos_model_path).resolve()
        if not (path / "config.json").is_file():
            raise ValueError("Configure a local Cosmos checkpoint directory first.")
        device = self.settings.cosmos_device
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        dtype = torch.float32 if device == "cpu" else torch.float16
        processor = AutoProcessor.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=False,
                dtype=dtype,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        self.processor, self.model = processor, model
        self.provenance = {
            "backend": "transformers",
            "device": device,
            "dtype": str(dtype),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
            "checkpoint_identity": "Local checkpoint; config hash is not a weight checksum.",
        }

    def generate(self, messages: list[dict], *, max_tokens: int | None = None) -> str:
        """JSON-only output cannot be enforced natively; callers validate the text."""
        with self.lock:
            return self._generate(messages, max_tokens)

    def _generate(self, messages: list[dict], max_tokens: int | None) -> str:
        import torch

        self._load()
        images = []
        native = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                converted = []
                for item in content:
                    if item["type"] == "image_url":
                        url = item["image_url"]["url"]
                        prefix = "data:image/jpeg;base64,"
                        if not url.startswith(prefix):
                            raise ValueError("Only supplied inline JPEG evidence is supported.")
                        with Image.open(
                            io.BytesIO(base64.b64decode(url[len(prefix) :], validate=True))
                        ) as image:
                            images.append(image.convert("RGB"))
                        converted.append({"type": "image"})
                    else:
                        converted.append(item)
                content = converted
            native.append({"role": message["role"], "content": content})
        text = self.processor.apply_chat_template(
            native, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(
            self.model.device
        )
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=max_tokens or self.settings.cosmos_max_new_tokens,
                max_time=self.settings.provider_timeout_seconds,
            )
        return self.processor.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
