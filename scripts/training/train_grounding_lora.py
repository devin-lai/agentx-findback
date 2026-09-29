"""LoRA fine-tuning of a Qwen3-VL-family checkpoint (Cosmos-Reason2) for reference-crop localization.

Each training example is rendered with the application's own `grounding_system` and
`grounding_messages`, converted to the Hugging Face processor format, and scored only on the
answer tokens (the JSON object plus the end-of-turn token). The vision tower stays frozen;
LoRA adapts the language model's attention and MLP projections.

Runs on the DGX Spark GB10 in bf16 with gradient checkpointing, one example per micro-batch
(image token counts differ per example) and gradient accumulation.
"""

import argparse
import hashlib
import io
import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from PIL import Image

from agentx.agents.providers import grounding_messages, grounding_system
from agentx.agents.skills import SkillRegistry

END_TOKEN = "<|im_end|>\n"
LANGUAGE_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def hf_messages(messages: list[dict]) -> tuple[list[dict], list[bytes]]:
    """OpenAI-style parts -> processor parts; images returned in prompt order."""
    images: list[bytes] = []
    converted = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            converted.append({"role": message["role"], "content": content})
            continue
        parts = []
        for part in content:
            if part["type"] == "image_url":
                import base64

                images.append(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))
                parts.append({"type": "image"})
            else:
                parts.append({"type": "text", "text": part["text"]})
        converted.append({"role": message["role"], "content": parts})
    return converted, images


def encode(processor, system: str, row: dict, root: Path, end_token: str):
    reference = (root / row["reference"]).read_bytes()
    picture = (root / row["image"]).read_bytes()
    messages = grounding_messages(
        system,
        picture,
        {"id": 0, "at_ms": 0},
        row["target"],
        reference,
        integer_points=True,
    )
    converted, images = hf_messages(messages)
    prompt = processor.apply_chat_template(converted, tokenize=False, add_generation_prompt=True)
    pil = [Image.open(io.BytesIO(data)).convert("RGB") for data in images]
    full = prompt + row["answer"] + end_token
    batch = processor(text=[full], images=pil, return_tensors="pt")
    prompt_ids = processor(text=[prompt], images=pil, return_tensors="pt")["input_ids"]
    labels = batch["input_ids"].clone()
    labels[:, : prompt_ids.shape[1]] = -100
    if not torch.equal(batch["input_ids"][:, : prompt_ids.shape[1]], prompt_ids):
        raise ValueError("Prompt tokens are not a prefix of the full example.")
    batch["labels"] = labels
    return batch


def collate(batches: list[dict], pad_id: int) -> dict:
    """Right-pad token tensors; concatenate image tensors in sample order (the model consumes
    image tokens and `image_grid_thw` rows in the same flattened order)."""
    width = max(b["input_ids"].shape[1] for b in batches)

    def pad(key: str, value: int) -> torch.Tensor:
        return torch.cat(
            [
                torch.nn.functional.pad(b[key], (0, width - b[key].shape[1]), value=value)
                for b in batches
            ]
        )

    return {
        "input_ids": pad("input_ids", pad_id),
        "attention_mask": pad("attention_mask", 0),
        "labels": pad("labels", -100),
        "pixel_values": torch.cat([b["pixel_values"] for b in batches]),
        "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batches]),
    }


class EncodedGroups(torch.utils.data.Dataset):
    """Encodes micro-batches in DataLoader workers so CPU preprocessing overlaps GPU steps."""

    def __init__(self, model: Path, system: str, rows: list, groups: list, root: Path):
        self.model, self.system, self.rows, self.groups, self.root = (
            model,
            system,
            rows,
            groups,
            root,
        )
        self.processor = None

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict:
        if self.processor is None:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model)
        encoded = [
            encode(self.processor, self.system, self.rows[i], self.root, END_TOKEN)
            for i in self.groups[index]
        ]
        pad_id = self.processor.tokenizer.pad_token_id
        return collate(encoded, pad_id) if len(encoded) > 1 else encoded[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="Directory with samples.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1, help="Examples per micro-batch")
    parser.add_argument(
        "--no-checkpointing", action="store_true", help="Trade memory for ~30% less compute"
    )
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--save-every", type=int, default=250, help="Optimizer steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6, help="CPU encoding workers")
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.55,
        help="Cap on this process's share of GB10 unified memory. Without it an oversized batch "
        "does not raise CUDA OOM: it starves the operating system and the node stops responding.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing training run.")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    args.output.mkdir(parents=True)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = [json.loads(line) for line in (args.data / "samples.jsonl").read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    skill, trace = SkillRegistry().load("review-visual-evidence")
    system = grounding_system([skill], integer_points=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda"}
    )
    if not args.no_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=LANGUAGE_TARGETS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_examples = math.ceil(len(rows) * args.epochs)
    total_micro = math.ceil(total_examples / args.batch_size)
    total_steps = math.ceil(total_micro / args.accumulation)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    warmup = max(1, int(total_steps * args.warmup))

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    run = {
        "started_at": datetime.now(UTC).isoformat(),
        "model": str(args.model),
        "data_manifest_sha256": hashlib.sha256(
            (args.data / "manifest.json").read_bytes()
        ).hexdigest(),
        "samples": len(rows),
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "skill": trace.model_dump(),
        "trainable_parameters": trainable,
        "args": {k: str(v) for k, v in vars(args).items()},
        "optimizer_steps": total_steps,
        "log": [],
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps({k: run[k] for k in ("samples", "trainable_parameters", "optimizer_steps")}))
    model.train()
    order = []
    while len(order) < total_examples:
        epoch = list(range(len(rows)))
        random.shuffle(epoch)
        order += epoch
    order = order[:total_examples]
    groups = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)]
    loader = torch.utils.data.DataLoader(
        EncodedGroups(args.model, system, rows, groups, args.data),
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        prefetch_factor=4 if args.workers else None,
        persistent_workers=bool(args.workers),
    )
    step, running, tokens, started = 0, 0.0, 0, time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for micro, batch in enumerate(loader, start=1):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        tokens += int(batch["attention_mask"].sum())
        loss = model(**batch).loss / args.accumulation
        loss.backward()
        running += float(loss.detach()) * args.accumulation
        if micro % args.accumulation == 0 or micro == len(groups):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total_steps:
                elapsed = time.monotonic() - started
                entry = {
                    "step": step,
                    "micro": micro,
                    "loss": round(running / (10 * args.accumulation), 5)
                    if step % 10 == 0
                    else None,
                    "lr": lr_at(step - 1),
                    "seconds": round(elapsed, 1),
                    "samples_per_second": round(micro * args.batch_size / elapsed, 3),
                    "tokens_per_second": round(tokens / elapsed, 1),
                }
                run["log"].append(entry)
                print(json.dumps(entry), flush=True)
                running = 0.0
                (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
            if step % args.save_every == 0:
                model.save_pretrained(args.output / f"checkpoint-{step}")
    model.save_pretrained(args.output / "adapter")
    processor.save_pretrained(args.output / "adapter")
    run["finished_at"] = datetime.now(UTC).isoformat()
    run["train_seconds"] = round(time.monotonic() - started, 1)
    run["peak_memory_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(
        json.dumps(
            {"done": True, "seconds": run["train_seconds"], "peak_gib": run["peak_memory_gib"]}
        )
    )


if __name__ == "__main__":
    main()
