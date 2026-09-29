# Skill benchmark: FindBack video memory

This team-authored Agent Skill was compared with no Skill on the same prompts and generated FindBack server. The published comparison covers two coding agents and a smaller local model. It is not an NVIDIA evaluation or a natural-user accuracy study.

## Results at a Glance

| Population | Without Skill | With Skill | Limit |
| --- | ---: | ---: | --- |
| Held out positive prompts, coding agent A | 1/15 | **15/15** | Three small sets |
| Held out positive prompts, coding agent B | 1/15 | **15/15** | Three small sets |
| Development prompts, coding agent A | 6/18 | **17/18** | Inspected during development |
| Development prompts, coding agent B | 8/18 | **16/18** | Inspected during development |
| Held out positive prompts, smaller local model | 2/16 | **5/16** | It also activated on 7/9 negative prompts |

One coding agent activated the Skill on one of 15 negative prompts, then correctly refused to claim an object's live location. The other had no false activation in that set. The local model gained less and often used the Skill on unrelated requests. No token value appeared in the measured transcripts, and read and ask access prevented the agent from changing server memory.

The result supports the Skill's usefulness for these coding agents on generated fixtures. It does not establish that all agents will discover it, that a recording reveals a present physical location, or that the underlying tracker is accurate on arbitrary footage. The [competition results](../../docs/benchmarks/INDEX.md) keep these boundaries visible.

The published `evals/evals.json` (`sha256:537f8bca02217e4739e1598e12341f9a7cd50f10846438a4c4892018b513a265`) provides positive and negative routing examples for package checks. Full row level runs and experiment metadata are retained privately.
