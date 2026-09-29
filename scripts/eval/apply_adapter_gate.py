"""Apply the grounding adapter's frozen promotion gate to one evaluation session's reports.

The gate (frozen before training; summarized in docs/benchmarks/INDEX.md): promote the adapter
only if, against its base model re-run in the same session,

1. held-out visible point-in-box recall rises by at least 5 percentage points,
2. held-out precision is not lower, and
3. false "visible" claims on the development cohorts' absent requests do not increase.

Every count is recomputed from per-frame rows, and both arms must cover exactly the same frames
with the same labels, or the script refuses. The exact McNemar test on held-out visible frames
is reported for information; it is not part of the gate.
"""

import argparse
import json
import math
from pathlib import Path

from summarize_vlm_bakeoff import rows_by_key, score


def load(path: Path) -> dict:
    return rows_by_key(json.loads(path.read_text()))


def paired(base: dict, adapter: dict, cohort: str) -> None:
    if base.keys() != adapter.keys():
        raise SystemExit(f"{cohort}: the arms do not cover the same frames.")
    for key, row in base.items():
        if row["truth_visible"] != adapter[key]["truth_visible"]:
            raise SystemExit(f"{cohort}: labels differ at {key}.")


def mcnemar(base: dict, adapter: dict) -> dict:
    """Exact two-sided McNemar on visible frames: correct point-in-box, base vs adapter."""
    keys = [k for k, r in base.items() if r["truth_visible"]]
    gains = sum(
        adapter[k]["point_inside_target_box"] and not base[k]["point_inside_target_box"]
        for k in keys
    )
    losses = sum(
        base[k]["point_inside_target_box"] and not adapter[k]["point_inside_target_box"]
        for k in keys
    )
    n = gains + losses
    tail = sum(math.comb(n, i) for i in range(min(gains, losses) + 1)) / 2**n if n else 1.0
    return {
        "adapter_only_correct": gains,
        "base_only_correct": losses,
        "p_two_sided": min(1.0, 2 * tail),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-heldout", type=Path, required=True)
    parser.add_argument("--adapter-heldout", type=Path, required=True)
    parser.add_argument("--base-dev", type=Path, action="append", required=True)
    parser.add_argument("--adapter-dev", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.base_dev) != len(args.adapter_dev):
        raise SystemExit("Pass one adapter development report per base development report.")
    base_held, adapter_held = load(args.base_heldout), load(args.adapter_heldout)
    paired(base_held, adapter_held, "held-out")
    held = {"base": score(list(base_held.values())), "adapter": score(list(adapter_held.values()))}
    dev_base, dev_adapter, dev_cohorts = [], [], {}
    for base_path, adapter_path in zip(args.base_dev, args.adapter_dev, strict=True):
        base, adapter = load(base_path), load(adapter_path)
        paired(base, adapter, base_path.name)
        dev_base += base.values()
        dev_adapter += adapter.values()
        dev_cohorts[base_path.name] = {
            "base": score(list(base.values())),
            "adapter": score(list(adapter.values())),
        }
    dev = {"base": score(dev_base), "adapter": score(dev_adapter)}
    base, adapter = held["base"], held["adapter"]
    recall = {
        arm: s["correct_point_in_box"] / s["visible"] if s["visible"] else 0.0
        for arm, s in held.items()
    }
    checks = {
        "heldout_recall_gain_pp": round(100 * (recall["adapter"] - recall["base"]), 2),
        # Integer comparisons, so no rounded ratio can decide a boundary case.
        "heldout_recall_gain_at_least_5pp": 100
        * (adapter["correct_point_in_box"] * base["visible"])
        >= (100 * base["correct_point_in_box"] + 5 * base["visible"]) * adapter["visible"],
        "heldout_precision_not_lower": adapter["positive_claims"] > 0
        and adapter["correct_point_in_box"] * base["positive_claims"]
        >= base["correct_point_in_box"] * adapter["positive_claims"],
        "dev_false_visible_not_higher": dev["adapter"]["false_visible"]
        <= dev["base"]["false_visible"],
    }
    decision = all(v for k, v in checks.items() if isinstance(v, bool))
    result = {
        "gate": "grounding adapter promotion gate",
        "reports": {
            "base_heldout": str(args.base_heldout),
            "adapter_heldout": str(args.adapter_heldout),
            "base_dev": [str(p) for p in args.base_dev],
            "adapter_dev": [str(p) for p in args.adapter_dev],
        },
        "heldout": held,
        "heldout_mcnemar_visible_frames": mcnemar(base_held, adapter_held),
        "development": dev,
        "development_by_cohort": dev_cohorts,
        "checks": checks,
        "promote": decision,
    }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
