import hashlib
import json
import os
import sys
from pathlib import Path

import gradio as gr


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from elsst_baselines.evaluator import data, track1, track2
from elsst_baselines.evaluator.leaderboard import LeaderboardStore, RateLimitError
from elsst_baselines.evaluator.validation import SubmissionValidationError


TRACK_LABELS = {
    "track1": "Track1 Retrieval",
    "track2": "Track2 Generation",
}


def _leaderboard_db_path():
    configured = os.environ.get("ELSST_LEADERBOARD_DB")
    if configured:
        return Path(configured)
    data_dir = Path("/data")
    if data_dir.exists() and os.access(data_dir, os.W_OK):
        return data_dir / "elsst_leaderboard.sqlite"
    return REPO_ROOT / "elsst_leaderboard.sqlite"


def _leaderboard_store():
    return LeaderboardStore(_leaderboard_db_path())


def _uploaded_path(uploaded_file):
    if uploaded_file is None:
        raise gr.Error("Upload a JSONL submission file.")
    if isinstance(uploaded_file, (str, Path)):
        return Path(uploaded_file)
    if hasattr(uploaded_file, "name"):
        return Path(uploaded_file.name)
    raise gr.Error("Unsupported uploaded file object.")


def _submission_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate(track, split, submission_file):
    submission_path = _uploaded_path(submission_file)
    if track == "track1":
        reference_rows = data.load_track1_reference(split)
        concept_pool = data.load_track1_concept_pool()
        return track1.score_submission(
            submission_path=submission_path,
            reference_rows=reference_rows,
            concept_pool=concept_pool,
            split=split,
        )
    if track == "track2":
        reference_rows = data.load_track2_reference(split)
        return track2.score_submission(
            submission_path=submission_path,
            reference_rows=reference_rows,
            split=split,
        )
    raise gr.Error(f"Unsupported track: {track}")


def _format_float(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _format_result(result):
    metric_rows = "\n".join(
        f"| `{name}` | {_format_float(value)} |"
        for name, value in result.metrics.items()
    )
    diagnostic_rows = "\n".join(
        f"| `{name}` | `{json.dumps(value, ensure_ascii=False)}` |"
        for name, value in result.diagnostics.items()
    )
    return (
        f"### {TRACK_LABELS[result.track]} {result.split} score\n\n"
        f"Primary metric: `{result.primary_metric}` = "
        f"`{_format_float(result.metrics[result.primary_metric])}`\n\n"
        "| Metric | Value |\n| --- | ---: |\n"
        f"{metric_rows}\n\n"
        "| Diagnostic | Value |\n| --- | --- |\n"
        f"{diagnostic_rows}"
    )


def _format_validation_error(exc):
    rows = "\n".join(f"- {error}" for error in exc.errors)
    return f"### Validation failed\n\n{rows}"


def _hf_username(request):
    if request is None:
        return None
    username = getattr(request, "username", None)
    if username:
        return username
    profile = getattr(request, "oauth_profile", None)
    if isinstance(profile, dict):
        return profile.get("preferred_username") or profile.get("name")
    headers = getattr(request, "headers", None)
    if headers is not None:
        username = headers.get("x-hf-username") or headers.get("X-HF-Username")
        if username:
            return username
        authorization = headers.get("authorization") or headers.get("Authorization")
        if isinstance(authorization, str) and authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1].strip()
            if token:
                try:
                    from huggingface_hub import HfApi

                    payload = HfApi(token=token).whoami()
                    return payload.get("name")
                except Exception:
                    return None
    return None


def score_val_file(track, submission_file):
    result = _evaluate(track, "val", submission_file)
    return _format_result(result)


def _score_val_ui(track, submission_file):
    try:
        result = _evaluate(track, "val", submission_file)
    except SubmissionValidationError as exc:
        return _format_validation_error(exc), {"validated": False, "track": track}, gr.update(interactive=False)
    except FileNotFoundError as exc:
        raise gr.Error(str(exc)) from exc
    return _format_result(result), {"validated": True, "track": track}, gr.update(interactive=True)


def _leaderboard_rows(track):
    entries = _leaderboard_store().top_entries(track=track)
    rows = []
    for rank, entry in enumerate(entries, start=1):
        rows.append(
            [
                rank,
                entry["username"],
                entry["model_name"],
                _format_float(entry["primary_score"]),
                entry["primary_metric"],
                entry["created_at"],
                entry["submission_hash"][:16],
            ]
        )
    return rows


def submit_test_file(track, model_name, submission_file, username):
    if not username:
        raise gr.Error("HF login is required for test submissions.")
    result = _evaluate(track, "test", submission_file)
    submission_hash = _submission_hash(_uploaded_path(submission_file))
    _leaderboard_store().record_submission(
        username=username,
        track=track,
        model_name=model_name,
        primary_metric=result.primary_metric,
        metrics=result.metrics,
        submission_hash=submission_hash,
    )
    return _format_result(result)


def _submit_test_ui(track, model_name, submission_file, val_state, request: gr.Request):
    if not val_state or not val_state.get("validated") or val_state.get("track") != track:
        raise gr.Error("Validate the selected track on val before submitting test results.")
    username = _hf_username(request)
    if not username:
        raise gr.Error("HF login is required for test submissions.")
    try:
        result_markdown = submit_test_file(track, model_name, submission_file, username)
    except SubmissionValidationError as exc:
        return _format_validation_error(exc), _leaderboard_rows(track)
    except RateLimitError as exc:
        raise gr.Error(str(exc)) from exc
    except FileNotFoundError as exc:
        raise gr.Error(str(exc)) from exc
    return result_markdown, _leaderboard_rows(track)


def build_demo():
    with gr.Blocks(title="ELSST Evaluator") as demo:
        gr.Markdown("# ELSST Evaluator")
        gr.Markdown(
            "Score validation submissions anonymously. Sign in with Hugging Face before "
            "submitting hidden-test results to the leaderboard."
        )
        gr.LoginButton()
        val_state = gr.State({"validated": False, "track": "track1"})

        with gr.Row():
            track_selector = gr.Radio(
                choices=list(TRACK_LABELS.keys()),
                value="track1",
                label="Track",
            )
            model_name = gr.Textbox(label="Model or team", value="unnamed")

        with gr.Row():
            with gr.Column():
                val_file = gr.File(label="Validation submission", file_types=[".jsonl"], type="filepath")
                val_button = gr.Button("Score validation")
                val_output = gr.Markdown()
            with gr.Column():
                test_file = gr.File(
                    label="Test submission",
                    file_types=[".jsonl"],
                    type="filepath",
                    interactive=False,
                )
                test_button = gr.Button("Submit test")
                test_output = gr.Markdown()

        leaderboard = gr.Dataframe(
            headers=[
                "rank",
                "user",
                "model",
                "score",
                "metric",
                "submitted_at",
                "submission_hash",
            ],
            datatype=["number", "str", "str", "str", "str", "str", "str"],
            interactive=False,
            label="Leaderboard",
        )
        refresh_button = gr.Button("Refresh leaderboard")

        val_button.click(
            _score_val_ui,
            inputs=[track_selector, val_file],
            outputs=[val_output, val_state, test_file],
        )
        test_button.click(
            _submit_test_ui,
            inputs=[track_selector, model_name, test_file, val_state],
            outputs=[test_output, leaderboard],
        )
        refresh_button.click(
            _leaderboard_rows,
            inputs=[track_selector],
            outputs=[leaderboard],
        )
        track_selector.change(
            lambda track: ({"validated": False, "track": track}, gr.update(interactive=False), _leaderboard_rows(track)),
            inputs=[track_selector],
            outputs=[val_state, test_file, leaderboard],
        )
    return demo


if __name__ == "__main__":
    build_demo().queue(default_concurrency_limit=1).launch()
