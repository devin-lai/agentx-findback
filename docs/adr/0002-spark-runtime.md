# ADR 0002: NVIDIA models on the DGX Spark node, grounded visual review, identity embeddings

Status: Accepted. Supersedes the StepFun planning line of ADR 0001.

## Context

The assigned DGX Spark (GB10, 121 GB unified memory, CUDA 13.0) is the competition platform. Its network reaches only the Aliyun PyPI mirror, npmmirror and ModelScope. The organizer image already provides vLLM 0.28 and NVIDIA Nemotron 3.5 Lightning 30B-A3B as a ModelOpt mixed-precision NVFP4 checkpoint; the team selected and served it but did not quantize it. Measured on the node, Cosmos-Reason2 2B and 8B both answer the fixture's "where was it last visible" question wrongly when given eight frames at once, while 8B localizes the object correctly in every single frame.

## Decision

- **Planner:** the query workflow gains a bounded tool agent behind a generic OpenAI-compatible JSON-mode endpoint. On the node it is Nemotron served by vLLM on loopback (`scripts/spark/serve_nemotron.sh`). StepFun credentials remain accepted aliases; no hosted API is required to run the product.
- **Visual review:** Cosmos-Reason2-8B served by vLLM (`scripts/spark/serve_cosmos.sh`). The default review for a registered object is *grounded*: one localization per frame with the registration crop as reference, ordered and named in code. Free-form multi-image review remains for open questions and as the measured baseline.
- **Identity:** DINOv2-small embeddings decide which RT-DETR proposal is the registered object, enabling several objects per category and explicit ambiguity. Thresholds are configuration, recorded in run provenance.
- **Node environment:** a user-level venv installed through the Aliyun index (CUDA aarch64 wheels), static FFmpeg from `imageio-ffmpeg`, Cosmos checkpoints from ModelScope, small HuggingFace weights relayed into the offline cache. No host, driver or organizer environment changes.

## Consequences

The product runs fully local on NVIDIA hardware with NVIDIA models; nothing leaves the node. The agent's action space is fixed in code, so a larger model improves wording and tool choice but cannot create new capabilities or unverified evidence. Grounded review trades one multi-image call for up to eight concurrent single-image calls (about 9 s on the node) and only answers object-centric questions; free-form review stays available. DINOv2 similarity is a measurement, not calibrated identity; the margin rule prefers an explicit `identity_ambiguous` over a wrong assignment. The 2B checkpoint stays supported for lower-memory hosts but is not the default.
