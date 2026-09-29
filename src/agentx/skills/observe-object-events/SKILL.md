---
name: observe-object-events
description: Build causal object events from registered visual observations.
version: 1.1.0
---

Use when processing a registered video's observations. Process timestamps forward. Preserve every sampled observation separately from compact events. A predicted trajectory is not an observation. Confirm a zone transition with consecutive detections. Missing detections remove visible status immediately. Identity ambiguity requires an unknown state. A camera change does not remove an observation: keep the box measured in that frame, and name a region only while the frame can be related to the registration view. Bind events to their actual observation IDs. Never infer theft, a person's identity, or an unseen destination. Registering an object does not prove its later visibility.
