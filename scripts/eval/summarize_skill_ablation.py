"""Summarize a skill-ablation report: per-arm totals, paired test and every disagreement.

Reads the JSON written by `evaluate_skill_ablation.py`, including a `status=running`
checkpoint, and recomputes each figure from `rows[]` rather than trusting the stored summary.
Pass `--suite` to restrict the summary to one frozen suite.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

ARMS = ("local", "agent_no_skills", "agent_skills")


def mcnemar(rows: list[dict]) -> dict:
    keyed = {(r["suite"], r["number"]): r for r in rows if r["arm"] == "agent_no_skills"}
    b = c = 0
    flips = []
    for row in rows:
        if row["arm"] != "agent_skills":
            continue
        other = keyed.get((row["suite"], row["number"]))
        if other is None or row["passed"] == other["passed"]:
            continue
        if row["passed"]:
            b += 1
        else:
            c += 1
        flips.append(
            {
                "suite": row["suite"],
                "number": row["number"],
                "text": row["case"]["text"],
                "skills_passed": row["passed"],
                "skills_failed": [k for k, v in row["checks"].items() if not v],
                "no_skills_failed": [k for k, v in other["checks"].items() if not v],
            }
        )
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n))
    return {"b": b, "c": c, "n_discordant": n, "p_value": round(p, 6), "flips": flips}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--suite", help="Restrict to one suite stem.")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    rows = report["rows"]
    if args.suite:
        rows = [r for r in rows if r["suite"] == args.suite]
    if not rows:
        raise SystemExit("No rows matched.")

    print(f"report status: {report.get('status')}   rows: {len(rows)}")
    print(f"planner: {report.get('planner_model')}   config: {report.get('planner_config')}")
    header = f"{'arm':17}{'passed':>9}{'wrong obj':>11}{'fallback':>10}{'median s':>10}"
    print("\n" + header + f"{'p95 s':>8}{'prompt tok':>12}{'out tok':>9}")
    for arm in ARMS:
        subset = [r for r in rows if r["arm"] == arm]
        if not subset:
            continue
        seconds = sorted(r["seconds"] for r in subset)
        p95 = seconds[max(0, math.ceil(0.95 * len(seconds)) - 1)]
        print(
            f"{arm:17}"
            f"{str(sum(r['passed'] for r in subset)) + '/' + str(len(subset)):>9}"
            f"{sum(r['wrong_object'] for r in subset):>11}"
            f"{sum(r['fell_back'] for r in subset):>10}"
            f"{statistics.median(seconds):>10.2f}"
            f"{p95:>8.1f}"
            f"{sum(r['prompt_tokens'] for r in subset):>12}"
            f"{sum(r['completion_tokens'] for r in subset):>9}"
        )

    test = mcnemar(rows)
    print(
        f"\nexact McNemar, skills vs no-skills: b={test['b']} (skills only) "
        f"c={test['c']} (no-skills only) p={test['p_value']}"
    )
    for flip in test["flips"]:
        mark = "skills fixes " if flip["skills_passed"] else "skills breaks"
        print(f"  {mark} #{flip['suite']}:{flip['number']}  {flip['text'][:70]}")

    print("\nfailures per arm:")
    for arm in ARMS:
        bad = [r for r in rows if r["arm"] == arm and not r["passed"]]
        print(f"  {arm}: {len(bad)}")
        for row in bad:
            probe = row["case"].get("probe", "")
            failed = ",".join(k for k, v in row["checks"].items() if not v)
            print(
                f"     #{row['suite']}:{row['number']} [{failed}] {probe or row['case']['text'][:60]}"
            )

    print("\ncode guards that fired (occurrences across all answers, per planner arm):")
    guards = {
        "invalid selection plan rejected": lambda r: sum(
            1
            for t in r["answer"]["tools"]
            if t.get("name") == "find_objects" and str(t.get("result", "")).startswith("error:")
        ),
        "cited unretrieved or unsupported evidence": lambda r: sum(
            "cited evidence it never retrieved" in w for w in r["answer"]["warnings"]
        ),
        "answered about a different object": lambda r: sum(
            "answered about a different object" in w for w in r["answer"]["warnings"]
        ),
        "fell back to the deterministic resolver": lambda r: int(r["fell_back"]),
    }
    for arm in ("agent_no_skills", "agent_skills"):
        subset = [r for r in rows if r["arm"] == arm]
        if not subset:
            continue
        counts = {name: sum(fn(r) for r in subset) for name, fn in guards.items()}
        print(f"  {arm}: " + ", ".join(f"{name} {n}" for name, n in counts.items()))

    print("\nper suite (passed/cases):")
    for suite in sorted({r["suite"] for r in rows}):
        line = f"  {suite:36}"
        for arm in ARMS:
            subset = [r for r in rows if r["suite"] == suite and r["arm"] == arm]
            if subset:
                line += f" {arm}={sum(r['passed'] for r in subset)}/{len(subset)}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
