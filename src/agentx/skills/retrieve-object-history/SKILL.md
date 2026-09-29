---
name: retrieve-object-history
description: Retrieve the last observation or history of a registered object at a video cutoff.
version: 1.0.0
---

Use for location, last-seen and history questions. Resolve only against the supplied registered-object catalog. Return null when the requested object is not registered; never substitute an unrelated object. For multiple matching names ask the user to select one. Use the caller's cutoff, never a later timestamp from memory. The location intent asks for visibility at the cutoff; last_seen asks for the latest evidenced observation; history asks for ordered changes. Query structured observations and events. A match by name or similarity alone is not evidence of current location.
