# Skill benchmark: answer-with-memory-tools

This Skill was evaluated as part of the six-Skill FindBack prompt-contract ablation, not in isolation. The tested population was generated software fixtures with frozen labels; these numbers do not measure real-world object recognition.

## Package identity

- Skill: `answer-with-memory-tools`
- `SKILL.md` SHA-256: `57fc9d2218d33b29dd94397e675200708d6256b848848df804a338d69f70dde3`
- Routing tasks and negative triggers: `evals/evals.json`

## Joint result

Across 168 cases, the planner passed **163/168** without the six Skill bodies and **165/168** with them. The difference did not establish a held-out correctness gain: both planner arms passed **25/28** on the reserved suite. Rejected selection plans fell from **36 to 25** across all cases. Code guards retained source and citation constraints in both arms, with no wrong-object final answers in this study.

These outcomes cannot be assigned to this Skill individually. A one-Skill ablation would require a new frozen comparison. The [competition results](../../../../docs/benchmarks/INDEX.md) state the wider limitations.
