"""Summarize complete constructed planner regressions without hiding wrong selections."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(path: Path) -> dict:
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("status") != "complete":
        raise ValueError(f"Report is incomplete: {path}")
    rows = [row for row in report["rows"] if row["provider_requested"]]
    if not rows or len(rows) != len(report["dataset"]["cases"]):
        raise ValueError(f"Missing model-requested cases: {path}")
    if sorted(row["number"] for row in rows) != list(range(len(rows))):
        raise ValueError(f"Repeated or missing case numbers: {path}")
    for row in rows:
        if row["case"] != report["dataset"]["cases"][row["number"]]:
            raise ValueError(f"Case labels do not match the frozen dataset: {path}")
        if row["passed"] != all(row["checks"].values()):
            raise ValueError(f"Inconsistent pass flag: {path}")
    return {
        "report": str(path),
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset": report["dataset"]["name"],
        "cases_sha256": report["cases_sha256"],
        "cases": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "actual_model_passed": sum(
            row["passed"] and row["answer"]["planner"] == "agent" for row in rows
        ),
        "fallbacks": sum(row["answer"]["planner"] != "agent" for row in rows),
        "wrong_selections": sum(
            bool(row["answer"]["states"])
            and [state["name"] for state in row["answer"]["states"]]
            != [row["case"]["expected_object"]]
            for row in rows
        ),
        "missed_unique_matches": sum(
            bool(row["case"]["expected_object"]) and not row["answer"]["states"] for row in rows
        ),
        "notes": (
            "Constructed fixture questions; not natural-user or visual accuracy. "
            "Passing fallback answers remain separate from actual model passes. "
            "Wrong selections include selecting any object when the gold label requires none. "
            "Other failures can be intent, none-versus-ambiguous resolution or evidence contracts. "
            "Observed zero wrong selections is not a general guarantee."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [summarize(path) for path in args.reports]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"reports": rows}, indent=2) + "\n")
    print(
        "| Report | Application pass | Actual model pass | Wrong selections | Missed unique | Fallbacks |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {Path(row['report']).stem} | {row['passed']}/{row['cases']} | "
            f"{row['actual_model_passed']}/{row['cases']} | {row['wrong_selections']} | "
            f"{row['missed_unique_matches']} | {row['fallbacks']} |"
        )


if __name__ == "__main__":
    main()
