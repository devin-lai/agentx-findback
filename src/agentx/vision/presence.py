"""A causal presence guard for a tracker that can follow a full occluder.

Presence logits are uncalibrated model scores. A lower-scoring mask needs a match
to a previously accepted appearance. Rejected views never update the bank.
"""

import numpy as np


class PresenceGate:
    def __init__(self, minimum: float, confident: float, similarity: float, recent: int = 8):
        self.minimum = minimum
        self.confident = confident
        self.similarity = similarity
        self.recent = recent
        self.banks: dict[int, list[np.ndarray]] = {}

    def seed(self, object_id: int, embedding: np.ndarray) -> None:
        self.banks[object_id] = [embedding]

    def admit(
        self, object_id: int, presence: float, embedding: np.ndarray | None
    ) -> tuple[bool, float | None]:
        bank = self.banks.get(object_id, [])
        similarity = (
            float((np.stack(bank) @ embedding).max()) if bank and embedding is not None else None
        )
        accepted = presence >= self.minimum and (
            presence >= self.confident or (similarity is not None and similarity >= self.similarity)
        )
        if accepted and embedding is not None:
            if not bank:
                raise ValueError("Seed the appearance gate from the registration frame first.")
            bank.append(embedding)
            bank[:] = [bank[0], *bank[1:][-self.recent :]]
        return accepted, similarity
