---
name: verify-location-answer
description: Check that an answer describes only supported visual facts within the requested time.
version: 1.0.0
---

Use before answering a location/history request. Evidence must belong to the requested video, run and object, and its timestamp must not exceed the query cutoff. Missing source media invalidates a visual claim. Distinguish visible, last_seen, unknown and not_observed. A stored recording cannot establish a real-world current position. Include the observed time and evidence link. Do not infer a hidden container from disappearance. Do not output a percentage as calibrated certainty. End with uncertainty when there is no supported answer; do not invent a successful model review.
