# Skill benchmark: compile-object-selection

This Skill was evaluated as part of the six-Skill FindBack prompt-contract ablation, not in isolation. The tested population was generated software fixtures with frozen labels; these numbers do not measure real-world object recognition.

## Package identity

- Skill: `compile-object-selection`
- `SKILL.md` SHA-256: `7b8d6be2dbdb682581bb18c63ccdcbb06c4212b856f76e9f94428c2a1744e067`
- Routing tasks and negative triggers: `evals/evals.json`

## Joint result

Across 168 cases, the planner passed **163/168** without the six Skill bodies and **165/168** with them. The difference did not establish a held-out correctness gain: both planner arms passed **25/28** on the reserved suite. Rejected selection plans fell from **36 to 25** across all cases. Code guards retained source and citation constraints in both arms, with no wrong-object final answers in this study.

These outcomes cannot be assigned to this Skill individually. A one-Skill ablation would require a new frozen comparison. The [competition results](../../../../docs/benchmarks/INDEX.md) state the wider limitations.
