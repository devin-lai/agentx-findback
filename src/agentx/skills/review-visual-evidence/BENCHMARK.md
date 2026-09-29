# Skill benchmark: review-visual-evidence

This Skill was evaluated as part of the six-Skill FindBack prompt-contract ablation, not in isolation. The tested population was generated software fixtures with frozen labels; these numbers do not measure real-world object recognition.

## Package identity

- Skill: `review-visual-evidence`
- `SKILL.md` SHA-256: `33e47958778a2f68d4f834ee78cf5e994b1d67a43ded9fb4daf41f8e36e9631d`
- Routing tasks and negative triggers: `evals/evals.json`

## Joint result

Across 168 cases, the planner passed **163/168** without the six Skill bodies and **165/168** with them. The difference did not establish a held-out correctness gain: both planner arms passed **25/28** on the reserved suite. Rejected selection plans fell from **36 to 25** across all cases. Code guards retained source and citation constraints in both arms, with no wrong-object final answers in this study.

These outcomes cannot be assigned to this Skill individually. A one-Skill ablation would require a new frozen comparison. The [competition results](../../../../docs/benchmarks/INDEX.md) state the wider limitations.
