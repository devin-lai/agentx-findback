import hashlib
import re
from pathlib import Path

from agentx.domain.contracts import SkillTrace

# The frontmatter is a fixed, hand-written `key: value` block. Parsing it directly keeps the
# core free of a YAML dependency that is only ever available transitively here.
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
FIELD = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):[ \t]*(.*)$")


class SkillRegistry:
    NAMES = {
        "compile-object-selection": "Translate complete object criteria into a bounded memory query without guessing an identity.",
        "review-visual-evidence": "Compare bounded source frames and cite observable changes with uncertainty.",
        "observe-object-events": "Turn causal observations into evidence-linked changes.",
        "retrieve-object-history": "Resolve a registered object and retrieve its history up to a cutoff.",
        "verify-location-answer": "Validate identity, time boundaries and evidence before answering.",
        "answer-with-memory-tools": "Drive the tool agent: resolve the object, read state/history through tools, cite retrieved IDs.",
    }

    def __init__(self, root: Path | None = None):
        self.root = root or Path(__file__).parents[1] / "skills"

    def load(self, name: str) -> tuple[str, SkillTrace]:
        if name not in self.NAMES:
            raise ValueError("Unknown skill.")
        body = (self.root / name / "SKILL.md").read_text()
        return body, SkillTrace(
            name=name, sha256=hashlib.sha256(body.encode()).hexdigest(), purpose=self.NAMES[name]
        )

    def describe(self, name: str) -> dict:
        """One Skill's declared metadata, its hash and the exact text sent to a model.

        Publishing the body is deliberate: a Skill is a contract, and an answer's stored
        `skills[].sha256` is only auditable if the text behind it can be read back.
        """
        body, trace = self.load(name)
        header = FRONTMATTER.match(body)
        declared = {}
        if header:
            for line in header.group(1).splitlines():
                field = FIELD.match(line.strip())
                if field:
                    declared[field[1]] = field[2].strip()
        return {
            "name": name,
            "version": declared.get("version"),
            "description": declared.get("description"),
            "purpose": self.NAMES[name],
            "sha256": trace.sha256,
            "body": body,
        }

    def catalog(self) -> list[dict]:
        return [self.describe(name) for name in sorted(self.NAMES)]
