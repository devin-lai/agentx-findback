import logging
import time

from agentx.agents.agent import AgentFailure
from agentx.agents.resolution import direct_objects, lookup_question, requested_intent
from agentx.domain.contracts import Question
from agentx.services.answering import AMBIGUOUS_MATCH, NO_MATCH, grounded_answer
from agentx.services.live import LIVE_WARNING
from agentx.storage.models import QueryRecord, uid

logger = logging.getLogger(__name__)

AGENT_SKILLS = ("retrieve-object-history", "verify-location-answer", "answer-with-memory-tools")
LOCAL_SKILLS = ("retrieve-object-history", "verify-location-answer")
SELECTION_SKILLS = ("compile-object-selection",)


class QueryWorkflow:
    """Bounded planning -> retrieval -> validation. Models never mutate observed facts."""

    def __init__(self, database, memory, skills, settings, agent=None):
        self.db, self.memory, self.skills, self.settings = database, memory, skills, settings
        self.agent = agent

    def answer(self, run_id: str, question: Question) -> dict:
        started = time.monotonic()
        run, video, objects, cutoff = self.memory.context(run_id, question.at_ms)
        text = question.text.strip()
        if not text:
            raise ValueError("Enter a question.")
        if question.object_id and all(o.id != question.object_id for o in objects):
            raise LookupError("Selected object does not belong to this run.")
        use_agent = bool(
            question.use_provider and self.settings.planner_available and self.agent is not None
        )
        skill_bodies, traces = [], []
        lookup = lookup_question(text)
        # Keep the selection policy in the trace when a query contains an answer
        # directive, even if removing that directive leaves a direct name lookup.
        selection = (
            use_agent
            and not question.object_id
            and (lookup != text or not direct_objects(lookup, objects, cutoff))
        )
        active_skills = (
            SELECTION_SKILLS if selection else AGENT_SKILLS if use_agent else LOCAL_SKILLS
        )
        skill_names = active_skills if question.use_skills else ()
        for name in skill_names:
            body, trace = self.skills.load(name)
            skill_bodies.append(body)
            traces.append(trace.model_dump())
        tools = [{"name": "load_skill", "result": t["name"]} for t in traces]
        planner = "local"
        planner_model = None
        warnings: list[str] = []
        result = None
        if use_agent:
            try:
                result = self.agent.answer(
                    run_id, question, run, video, objects, cutoff, skill_bodies
                )
                planner = "agent"
                planner_model = (
                    self.settings.planner_endpoint[2] if result.get("model_used", True) else None
                )
            except AgentFailure as exc:
                logger.warning("Planner agent gave up (%s); using the local query workflow", exc)
                warnings.append(
                    "The planner model was unavailable or returned an invalid plan; the local "
                    "query workflow answered instead."
                )
            except Exception:
                # A defect in the agent must stay visible while the user still gets an answer.
                logger.exception("Planner agent crashed; using the local query workflow")
                warnings.append(
                    "The planner agent failed with an internal error; the local query workflow "
                    "answered instead. Check the server log."
                )
        if result is None:
            result = self.local(run_id, question, run, video, objects, cutoff)
        if video.live_status == "recording":
            warnings.append(LIVE_WARNING)
        response = {
            "id": uid(),
            "run_id": run_id,
            "question": text,
            "intent": result["intent"],
            "as_of_ms": cutoff,
            "answer": result["answer"],
            "states": result["states"],
            "events": result["events"],
            "evidence": result["evidence"],
            "skills": traces,
            "tools": tools + result["tools"],
            "planner": planner,
            "planner_model": planner_model,
            "warnings": warnings + result["warnings"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        with self.db.session.begin() as session:
            session.add(
                QueryRecord(
                    id=response["id"],
                    run_id=run_id,
                    question=text,
                    cutoff_ms=cutoff,
                    response=response,
                )
            )
        return response

    def local(self, run_id: str, question: Question, run, video, objects, cutoff: int) -> dict:
        """Deterministic resolver: registered names/categories and three intents."""
        explicit = [o for o in objects if o.id == question.object_id] if question.object_id else []
        matches = direct_objects(lookup_question(question.text), objects, cutoff)
        intent = requested_intent(question)
        tools: list[dict] = []
        selected = explicit or matches
        if question.object_id and not explicit:
            raise LookupError("Selected object does not belong to this run.")
        result: dict
        if len(selected) != 1:
            if len(selected) > 1:
                tools.append(
                    {
                        "name": "resolve_object",
                        "result": f"ambiguous · {len(selected)} matching objects",
                        "matching_objects": [{"object_id": o.id, "name": o.name} for o in selected],
                    }
                )
            result = {
                "intent": intent,
                "answer": AMBIGUOUS_MATCH if len(selected) > 1 else NO_MATCH,
                "states": [],
                "events": [],
                "evidence": [],
            }
        else:
            obj = selected[0]
            result = grounded_answer(self.memory, run, video, obj, cutoff, intent)
            tools.append(
                {
                    "name": "get_state",
                    "object_id": obj.id,
                    "cutoff_ms": cutoff,
                    "result": result["states"][0]["status"],
                }
            )
            if result["events"]:
                tools.append(
                    {"name": "query_history", "count": len(result["events"]), "cutoff_ms": cutoff}
                )
        tools.append(
            {
                "name": "validate_answer",
                "cutoff_ms": cutoff,
                "evidence_count": len(result["evidence"]),
                "result": "rendered from verified memory",
            }
        )
        return {**result, "tools": tools, "warnings": []}
