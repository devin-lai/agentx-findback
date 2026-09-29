"""Check the assigned planner's JSON-schema support without changing app settings."""

import json
import time
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field

from agentx.config import Settings
from agentx.services.selection import Predicate, SelectionPredicate


class Plan(Predicate):
    intent: Literal["location", "last_seen", "history"]
    filters: list[SelectionPredicate] = Field(max_length=8)


def main():
    settings = Settings()
    base, key, model = settings.planner_endpoint
    schema = Plan.model_json_schema()
    prompt = (
        'Translate the question to memory selection filters. Return only {"intent":"location","filters":[...]}. Use an empty list only for unsupported criteria. Do not invent a name, region or time absent from the question. Regions: left,center,right. Cutoff 11000 ms. Inventory: Red toolkit registered 0 ms, Blue remote registered 0 ms. State kind checks status/zone at a specific at_ms; default cutoff. first_seen checks first positive region, not arbitrary starting time. seen kind counts positives in zone (outside_zone=true counts outside it); min_count=0,max_count=0 means never. event kind counts appeared,moved,lost,reappeared,ambiguous. registered kind uses start_ms/end_ms. Times from start, in milliseconds. Full schema: '
        + json.dumps(schema)
    )
    results = []
    for text in [
        "Where is the object that is visible?",
        "Show the history of the item that was visible at the start.",
        "Where is the object that never left the center area?",
    ]:
        started = time.monotonic()
        with httpx.Client(timeout=60) as client:
            response = client.post(
                base.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}"} if key else {},
                json=dict(
                    model=model,
                    messages=[dict(role="system", content=prompt), dict(role="user", content=text)],
                    temperature=0,
                    max_tokens=800,
                    chat_template_kwargs={"enable_thinking": False},
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "memory_selection",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                ),
            )
        row = dict(question=text, status=response.status_code, seconds=time.monotonic() - started)
        if response.is_success:
            raw = response.json()["choices"][0]["message"]["content"]
            row["plan"] = Plan.model_validate_json(raw).model_dump()
        else:
            row["error"] = response.text[:1000]
        results.append(row)
        print(json.dumps(row), flush=True)
    Path("artifacts/schema-probe.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
