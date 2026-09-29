---
name: review-visual-evidence
description: Compare a bounded set of original video frames and cite observable changes.
version: 1.0.0
---

Use only the supplied frames and their video timestamps. Identify objects by visible attributes, not a guessed identity or a label printed in the recording. Describe where an object was last visible and what changed between the earliest and latest relevant frames. Cite the frame IDs supporting each observation in evidence_frame_ids and mention the relevant video timestamps in the summary.

Separate an observation from a hypothesis. Sampled frames omit intermediate actions. A disappearance establishes absence from the sampled view, not placement inside a drawer or bag. A recording cannot establish a present real-world location. Similar-looking objects can be confused. If the target is absent or unclear, say so; do not invent a match. Preserve that uncertainty even when the question assumes an action occurred.

Return the required JSON schema without extra keys, percentages, bounding boxes, hidden reasoning or commands. This is a review for a person, not permission for a robot to act. The review never changes registered identities, events or authoritative memory.
