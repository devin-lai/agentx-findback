---
name: answer-with-memory-tools
description: Drive the FindBack tool agent. Resolve the object, read memory state and history through tools, cite retrieved observation IDs, and state uncertainty before finishing.
version: 2
---
# Answer with memory tools

1. Match the question to one registered object by name or category. Ambiguous matches: finish and ask the user to select one object; cite nothing.
2. For an indirect description, call `find_objects` with the complete typed criteria. Code checks every eligible object. Zero matches means no match; multiple matches require clarification. Never choose an arbitrary candidate, omit part of the description, or weaken a criterion to obtain a match. Times are measured from video start and cannot exceed the cutoff. Current visibility excludes last-seen states.
3. Always call `get_state` first for the matched object. Its `evidence_observation_id` is the only citation for a location or last-seen answer.
4. Call `get_history` when the question mentions history, movement, before/after, change, timeline or "what happened". Report only recorded `appeared`, `moved`, `reappeared`, `lost` and `ambiguous` events with their times and zone names.
5. Call `review_frames` at most once, only for a visual detail memory cannot hold (for example what is next to the object). Present its summary as a non-authoritative interpretation, never as the answer's location fact.
6. Status `visible` means the object was observed at the cutoff; `last_seen` means only a past sighting exists; `unknown` and `not_observed` mean no supported position. Never turn a past sighting into a present location or guess where an object went.
7. The final answer names the object, the supported time and zone in words, and the limits of the recording. Cite observation IDs exactly as returned by tools.
