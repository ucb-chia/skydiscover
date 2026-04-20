from dataclasses import dataclass, field
from typing import Any, Dict, Union


@dataclass
class EvaluationResult:
    """
    Result of program evaluation containing both metrics and optional artifacts
    """

    metrics: Dict[str, float]
    artifacts: Dict[str, Union[str, bytes]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Build an EvaluationResult from a dict.

        If `data` contains an `artifacts` key with a dict value, it is split
        out into the artifacts field. All other entries become metrics.
        Non-numeric metric values are kept as-is (callers may filter later).
        """
        if isinstance(data, dict) and isinstance(data.get("artifacts"), dict):
            artifacts = data["artifacts"]
            metrics = {k: v for k, v in data.items() if k != "artifacts"}
            return cls(metrics=metrics, artifacts=artifacts)
        return cls(metrics=dict(data))

    def to_dict(self) -> Dict[str, Any]:
        result = dict(self.metrics)
        if self.artifacts:
            result["artifacts"] = self.artifacts
        return result
