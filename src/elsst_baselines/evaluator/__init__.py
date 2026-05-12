"""Upload-based scorers and leaderboard storage for ELSST Spaces."""

from elsst_baselines.evaluator.result import EvaluationResult
from elsst_baselines.evaluator.validation import SubmissionValidationError

__all__ = ["EvaluationResult", "SubmissionValidationError", "track1", "track2"]
