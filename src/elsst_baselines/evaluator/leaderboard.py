import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Hugging Face Spaces run on Linux.
    fcntl = None


class RateLimitError(RuntimeError):
    pass


class LeaderboardStore:
    def __init__(self, db_path, daily_limit=3):
        self.db_path = Path(db_path)
        self.daily_limit = daily_limit
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _lock(self):
        lock_path = self.db_path.with_suffix(self.db_path.suffix + ".lock")
        with lock_path.open("w", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self):
        with self._lock():
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS submissions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        username TEXT NOT NULL,
                        track TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        primary_metric TEXT NOT NULL,
                        primary_score REAL NOT NULL,
                        metrics_json TEXT NOT NULL,
                        submission_hash TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_submissions_rate_limit
                    ON submissions (username, track, created_at)
                    """
                )
                connection.commit()

    def record_submission(
        self,
        username,
        track,
        model_name,
        primary_metric,
        metrics,
        submission_hash,
        now=None,
    ):
        username = str(username).strip()
        track = str(track).strip()
        model_name = str(model_name).strip() or "unnamed"
        if not username:
            raise ValueError("username is required")
        if not track:
            raise ValueError("track is required")
        if primary_metric not in metrics:
            raise ValueError(f"primary metric {primary_metric!r} missing from metrics")

        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        cutoff = now - timedelta(hours=24)
        created_at = now.astimezone(timezone.utc).isoformat()
        cutoff_at = cutoff.astimezone(timezone.utc).isoformat()

        with self._lock():
            with closing(self._connect()) as connection:
                if self.daily_limit is not None:
                    count = connection.execute(
                        """
                        SELECT COUNT(*) AS count
                        FROM submissions
                        WHERE username = ? AND track = ? AND created_at >= ?
                        """,
                        (username, track, cutoff_at),
                    ).fetchone()["count"]
                    if count >= self.daily_limit:
                        raise RateLimitError(
                            f"rate limit exceeded for {username} on {track}: "
                            f"{self.daily_limit} submissions per 24 hours"
                        )

                connection.execute(
                    """
                    INSERT INTO submissions (
                        created_at,
                        username,
                        track,
                        model_name,
                        primary_metric,
                        primary_score,
                        metrics_json,
                        submission_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        created_at,
                        username,
                        track,
                        model_name,
                        primary_metric,
                        float(metrics[primary_metric]),
                        json.dumps(metrics, ensure_ascii=False, sort_keys=True),
                        submission_hash,
                    ),
                )
                connection.commit()

    def top_entries(self, track=None, limit=100):
        query = "SELECT * FROM submissions"
        params = []
        if track is not None:
            query += " WHERE track = ?"
            params.append(track)
        query += " ORDER BY created_at DESC"

        with self._lock():
            with closing(self._connect()) as connection:
                rows = [dict(row) for row in connection.execute(query, params).fetchall()]

        best_by_key = {}
        for row in rows:
            key = (row["username"], row["track"], row["model_name"])
            current = best_by_key.get(key)
            if current is None or (
                row["primary_score"],
                row["created_at"],
            ) > (
                current["primary_score"],
                current["created_at"],
            ):
                row["metrics"] = json.loads(row.pop("metrics_json"))
                best_by_key[key] = row

        entries = sorted(
            best_by_key.values(),
            key=lambda item: (item["track"], -item["primary_score"], item["created_at"]),
        )
        return entries[:limit]
