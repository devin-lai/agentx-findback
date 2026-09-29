# Third-party notices

AgentX original source is licensed under Apache-2.0. This file identifies major components; it does not replace their complete licenses or the transitive dependency notices installed with them. Lock files identify the resolved dependency set. Retain upstream notices when distributing binaries or model artifacts.

Every frontend production build emits `web/dist/assets/THIRD_PARTY_LICENSES.txt` with the complete notices from the JavaScript, icon and font packages in its module graph. This file is included in the release bundle and Docker image and is served at `/assets/THIRD_PARTY_LICENSES.txt`. The Python wheel includes `LICENSE`, `NOTICE` and this component inventory under its distribution license directory.

| Component | Upstream | License / handling |
| --- | --- | --- |
| FastAPI | https://github.com/fastapi/fastapi | MIT |
| Pydantic | https://github.com/pydantic/pydantic | MIT |
| SQLAlchemy / Alembic | https://github.com/sqlalchemy | MIT |
| Svelte / Vite | https://github.com/sveltejs/svelte / https://github.com/vitejs/vite | MIT |
| Lucide icons | https://lucide.dev/license | ISC, with retained MIT notices for icons derived from Feather |
| DM Sans / Manrope | Fontsource packages in web/package-lock.json | SIL Open Font License 1.1; self-hosted package assets |
| OpenCV | https://github.com/opencv/opencv | Apache-2.0 for current OpenCV source; bundled codec components retain their notices |
| PyAV | https://github.com/PyAV-Org/PyAV | BSD-3-Clause; its FFmpeg-linked binary wheels have additional component notices. Live capture encodes H.264 in-process with the wheel's libx264 (GPL-2.0-or-later) when present, otherwise its OpenH264 (BSD-2-Clause, Cisco). Source archives do not bundle these codecs; a built Docker image includes installed dependencies and must retain their notices and applicable source obligations |
| FFmpeg | https://ffmpeg.org/legal.html | License depends on build configuration; this application invokes the installed executable, including its libx264 encoder |
| PyTorch | https://github.com/pytorch/pytorch | BSD-style license and bundled dependency notices |
| Transformers | https://github.com/huggingface/transformers | Apache-2.0 |
| RT-DETR R18 model | https://huggingface.co/PekingU/rtdetr_r18vd | Model card declares Apache-2.0; downloaded separately at the configured revision |
| Trackers / ByteTrack implementation | https://github.com/roboflow/trackers | Apache-2.0 |
| Supervision | https://github.com/roboflow/supervision | MIT |
| Cosmos Reason models | https://github.com/nvidia-cosmos/cosmos-reason2 | Model weights use NVIDIA Open Model License; code and weights are separate artifacts. Checkpoints were obtained from ModelScope (`nv-community/Cosmos-Reason2-2B`, `-8B`) on the competition node |
| NVIDIA Nemotron 3.5 Lightning 30B-A3B (NVFP4) | https://huggingface.co/nvidia | NVIDIA Open Model License; served locally with vLLM as the query planner; weights provided on the competition node, never committed |
| DINOv2 small | https://huggingface.co/facebook/dinov2-small | Apache-2.0; optional appearance-identity embeddings at the configured revision |
| SAM 2.1 Small (runtime) and Large (rejected research candidate) | https://github.com/facebookresearch/sam2 / https://huggingface.co/facebook/sam2.1-hiera-small | Apache-2.0 code and checkpoint; used through the existing Transformers video runtime; weights downloaded separately and hash-verified; the Large revision/hash manifest is retained under scripts/spark/manifests |
| LaSOT research footage and annotations | https://vision.cs.stonybrook.edu/~lasot/ / https://huggingface.co/datasets/l-lt/LaSOT | Local research benchmark only; toolkit licensing does not establish footage rights. Raw archives, prepared clips and image diagnostics are ignored artifacts and excluded from source releases |
| vLLM | https://github.com/vllm-project/vllm | Apache-2.0; separate inference process on the node, not an application dependency |
| imageio-ffmpeg | https://github.com/imageio/imageio-ffmpeg | BSD-2-Clause wrapper around a static FFmpeg build (GPL/LGPL components, including libx264); optional `ffmpeg` extra used only where no system FFmpeg exists |
| ModelScope | https://github.com/modelscope/modelscope | Apache-2.0 download client used on the node for Cosmos checkpoints |
| StepFun service | https://platform.stepfun.com/ | Hosted service; any OpenAI-compatible planner endpoint can be configured; API access and terms are separate from AgentX's source license |
| NVIDIA Cosmos3-Nano | ModelScope `nv-community/Cosmos3-Nano` | OpenMDW-1.1 (https://openmdw.ai/license/1-1/) per its model card; evaluated on the node as an alternative reviewer only; weights never committed |
| StepFun Step3-VL-10B | ModelScope `stepfun-ai/Step3-VL-10B` | Apache-2.0 per its model card; evaluated on the node as an alternative reviewer with a no-think copy of its chat template kept in the experiment directory; weights never committed |
| GOT-10k | http://got-10k.aitestunion.com/ / ModelScope `OpenDataLab/GOT-10k` | CC BY-NC-SA 4.0, research use; used on the node only to train the reviewer's LoRA adapter; neither the data nor the adapter weights are distributed by this repository, and anyone redistributing such weights must consider the dataset's non-commercial and share-alike terms |
| Pexels desk clips | https://www.pexels.com/license/ | Pexels License; six clips used for evaluation and the demo film; not redistributed source attribution and hashes are retained in private evaluation records |
| PEFT / Accelerate | https://github.com/huggingface/peft / https://github.com/huggingface/accelerate | Apache-2.0; training environment on the node only, not an application dependency |
| OpenSSF model-signing | https://github.com/sigstore/model-transparency | Apache-2.0; a separate environment signs and verifies the Skills |
| Playwright | https://github.com/microsoft/playwright | Apache-2.0; browser tests and the demo recorder (development dependency) |

No Cosmos, Cosmos3, Nemotron, Step3-VL, DINOv2, SAM2 or RT-DETR weights, and no LoRA adapter trained here, are committed or redistributed by AgentX. The local user-managed `model/` directory, if present, is ignored by Git. This project's Apache license does not relicense those weights. Review upstream attribution requirements before any separate model redistribution.

The generated fixture contains original procedural imagery and no downloaded videos, faces, private footage or prewritten model outputs. The composited evaluation clip used for the node measurements was built from three public-domain remote-control photographs on Wikimedia Commons (Microsoft Xbox 360 remote controls: black, white short, white long) placed on a procedural background; the photographs are not committed and the generator accepts any user-supplied images. `docs/images/findback-answer.png` shows the application over the licensed Pexels clip "Flatlay of School Materials" by MART PRODUCTION, one of the six desk clips listed above.
