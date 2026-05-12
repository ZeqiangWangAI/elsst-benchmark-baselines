from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    track: str
    split: str
    primary_metric: str
    metrics: dict
    diagnostics: dict
    valid: bool = True

    @property
    def primary_score(self):
        return self.metrics[self.primary_metric]
