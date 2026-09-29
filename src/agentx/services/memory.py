from sqlalchemy import select

from agentx.domain.contracts import Box, EvidenceRef, ObjectMemory, Visibility, timestamp
from agentx.storage.models import Event, IndexRun, Observation, RegisteredObject, Video

__all__ = ["Memory", "evidence_ref", "timestamp"]


def evidence_ref(observation, run, video, cutoff: int) -> EvidenceRef:
    return EvidenceRef(
        observation_id=observation.id,
        video_id=video.id,
        run_id=run.id,
        at_ms=observation.at_ms,
        box=Box.model_validate(observation.box),
        zone=observation.zone,
        scene_reference=observation.scene_reference,
        scene_transform=observation.scene_transform,
        frame_url=f"/api/v1/observations/{observation.id}/frame",
        media_url=f"/api/v1/videos/{video.id}/media#t={max(0, observation.at_ms / 1000 - 1):.3f},{min(cutoff, observation.at_ms + 1000) / 1000:.3f}",
    )


class Memory:
    def __init__(self, database, catalog):
        self.db, self.catalog = database, catalog

    def context(self, run_id: str, cutoff_ms: int | None):
        with self.db.session() as session:
            run = session.get(IndexRun, run_id)
            if run is None:
                raise LookupError("Analysis run not found.")
            if run.status not in {"running", "complete"}:
                raise ValueError("This run has no queryable memory. Build a completed index first.")
            video = session.get(Video, run.video_id)
            upper = video.duration_ms if run.status == "complete" else run.processed_ms
            cutoff = upper if cutoff_ms is None else cutoff_ms
            if cutoff < 0 or cutoff > upper:
                raise ValueError(
                    "That time has not been processed. Choose a time within the indexed range."
                )
            objects = session.scalars(
                select(RegisteredObject).where(RegisteredObject.id.in_(run.object_ids))
            ).all()
        return run, video, objects, cutoff

    def states(self, run_id: str, cutoff_ms: int | None = None) -> list[ObjectMemory]:
        run, video, objects, cutoff = self.context(run_id, cutoff_ms)
        return [self.object_state(run, video, obj, cutoff) for obj in objects]

    def object_state(self, run, video, obj, cutoff: int) -> ObjectMemory:
        with self.db.session() as session:
            criteria = [
                Observation.run_id == run.id,
                Observation.object_id == obj.id,
                Observation.at_ms <= cutoff,
            ]
            latest = session.scalar(
                select(Observation).where(*criteria).order_by(Observation.at_ms.desc()).limit(1)
            )
            last_seen = session.scalar(
                select(Observation)
                .where(*criteria, Observation.visible.is_(True))
                .order_by(Observation.at_ms.desc())
                .limit(1)
            )
        status = Visibility.NOT_OBSERVED
        reason = None
        evidence = None
        if last_seen:
            evidence = evidence_ref(last_seen, run, video, cutoff)
            status = Visibility.LAST_SEEN
        if latest:
            reason = latest.reason
            if latest.reason in {"identity_ambiguous", "identity_unconfirmed", "scene_changed"}:
                status = Visibility.UNKNOWN
            elif latest.visible and cutoff - latest.at_ms <= max(
                1000 / run.config["sample_fps"] * 1.5, 250
            ):
                status = Visibility.VISIBLE
            elif latest.visible:
                reason = "observation_expired"
        if evidence and (problem := self.catalog.source_problem(video)):
            status, reason, evidence = Visibility.UNKNOWN, problem, None
        return ObjectMemory(
            object_id=obj.id,
            name=obj.name,
            status=status,
            as_of_ms=cutoff,
            last_observed_ms=last_seen.at_ms if last_seen else None,
            zone=last_seen.zone if last_seen else None,
            current_zone=latest.zone if latest and status == Visibility.VISIBLE else None,
            reason=reason,
            evidence=evidence,
        )

    def camera_poses(self, run_id: str, start_ms: int, end_ms: int) -> list[dict]:
        """The camera evidence recorded while indexing, over a window of one run.

        One entry per sampled time, deduplicated across objects: the pose describes the frame, not
        the object. A caller uses it to read a position in registration coordinates without
        re-estimating anything.
        """
        with self.db.session() as session:
            rows = session.execute(
                select(Observation.at_ms, Observation.scene_reference, Observation.scene_transform)
                .where(
                    Observation.run_id == run_id,
                    Observation.at_ms >= start_ms,
                    Observation.at_ms <= end_ms,
                )
                .order_by(Observation.at_ms)
            ).all()
        poses: dict[int, dict] = {}
        for at_ms, reference, transform in rows:
            # A frame is `registered` for every object or for none, so first writer wins; keep the
            # one that carries a transform when several agree.
            current = poses.get(at_ms)
            if current is None or (current["scene_transform"] is None and transform is not None):
                poses[at_ms] = {
                    "at_ms": at_ms,
                    "scene_reference": reference,
                    "scene_transform": transform,
                }
        return list(poses.values())

    def history(
        self, run_id: str, cutoff_ms: int | None = None, object_id: str | None = None
    ) -> list[dict]:
        run, video, objects, cutoff = self.context(run_id, cutoff_ms)
        names = {o.id: o.name for o in objects}
        if object_id and object_id not in names:
            raise LookupError("Object does not belong to this run.")
        with self.db.session() as session:
            statement = select(Event).where(Event.run_id == run_id, Event.at_ms <= cutoff)
            if object_id:
                statement = statement.where(Event.object_id == object_id)
            events = session.scalars(statement.order_by(Event.at_ms, Event.id)).all()
            return [
                {
                    "id": e.id,
                    "object_id": e.object_id,
                    "name": names[e.object_id],
                    "at_ms": e.at_ms,
                    "kind": e.kind,
                    "zone": e.zone,
                    "previous_zone": e.previous_zone,
                    "reason": e.reason,
                    "observation_id": e.observation_id,
                }
                for e in events
            ]

    def evidence_for(self, observation_id: str, cutoff: int) -> EvidenceRef:
        """An evidence reference for a visible observation at or before the cutoff."""
        observation, run, video = self.observation(observation_id)
        if not observation.visible or observation.at_ms > cutoff:
            raise ValueError("That observation cannot serve as evidence at this cutoff.")
        self.catalog.require_source(video)
        return evidence_ref(observation, run, video, cutoff)

    def observation(self, observation_id: str):
        with self.db.session() as session:
            observation = session.get(Observation, observation_id)
            if observation is None:
                raise LookupError("Evidence not found.")
            run = session.get(IndexRun, observation.run_id)
            video = session.get(Video, run.video_id)
        return observation, run, video
