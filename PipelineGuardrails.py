import hashlib
import asyncio
import json
import os
import re
import secrets
import threading
import tempfile
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple


CRISIS_GUIDANCE = (
    "I can't continue with automated VR content generation because the information provided "
    "indicates a possible immediate safety risk. Please contact local emergency services or a "
    "local crisis service now, tell a trusted person who can stay with you, and move away from "
    "anything you could use to harm yourself. This system is a research prototype and is not an "
    "emergency or clinical service."
)

_HIGH_RISK_PATTERNS = (
    r"\b(?:kill|hurt) myself\b",
    r"\b(?:suicidal|suicide plan|end my life|want to die)\b",
    r"\bimmediate danger\b",
    r"自杀|结束生命|不想活|伤害自己|马上去死|立即危险",
)
_NEGATED_PATTERNS = (
    r"\b(?:not|never|no longer) suicidal\b",
    r"\b(?:do not|don't|never) want to (?:die|hurt myself|kill myself)\b",
    r"没有自杀(?:想法|意图)|不想自杀|不会伤害自己",
)
_JSONL_LOCK = threading.Lock()


class CrossProcessAsyncLock:
    """Serialize mutable GPU pipelines across endpoints and sibling server processes."""

    def __init__(self, path: str | None = None):
        self.path = Path(path or os.getenv(
            "GPU_LOCK_FILE", os.path.join(tempfile.gettempdir(), "vr-agent-gpu-generation.lock")
        ))
        self._local_lock = threading.Lock()
        self._handle = None

    def _acquire_file(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        self._handle.seek(0)
        if self._handle.read(1) == b"":
            self._handle.write(b"0")
            self._handle.flush()
        self._handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except BaseException:
            self._handle.close()
            self._handle = None
            raise

    def _release_file(self):
        if self._handle is None:
            return
        self._handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    def _acquire_all(self):
        self._local_lock.acquire()
        try:
            self._acquire_file()
        except BaseException:
            self._local_lock.release()
            raise

    def _release_all(self):
        try:
            self._release_file()
        finally:
            self._local_lock.release()

    async def __aenter__(self):
        acquisition = asyncio.create_task(asyncio.to_thread(self._acquire_all))
        try:
            await asyncio.shield(acquisition)
        except asyncio.CancelledError:
            # Cancellation cannot stop a thread waiting on an OS lock. Reap it,
            # then release ownership instead of leaving a ghost lock holder.
            await acquisition
            await asyncio.to_thread(self._release_all)
            raise
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        release = asyncio.create_task(asyncio.to_thread(self._release_all))
        try:
            await asyncio.shield(release)
        except asyncio.CancelledError:
            await release
            raise


def detect_risk_flags(user_text: str, psych_state: Dict[str, Any] | None) -> List[str]:
    """Combine model-produced flags with a conservative deterministic keyword backstop."""
    state = psych_state or {}
    raw_flags = state.get("risk_flags", []) or []
    if isinstance(raw_flags, str):
        raw_flags = [raw_flags]
    flags = [str(item).strip() for item in raw_flags if str(item).strip()]
    text = (user_text or "").strip().lower()
    if text and not any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _NEGATED_PATTERNS):
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _HIGH_RISK_PATTERNS):
            flags.append("deterministic_immediate_safety_keyword_match")
    return list(dict.fromkeys(flags))


def risk_blocked_result(session_id: str, mode: str, user_input: str, psych_state: Dict[str, Any], flags: List[str]):
    return {
        "session_id": session_id,
        "user_input": user_input,
        "psych_state": psych_state,
        "risk_flags": flags,
        "crisis_guidance": CRISIS_GUIDANCE,
        "intervention_plan": [],
        "music_playlist": [],
        "status": "blocked_risk",
        "generation_skipped": True,
        "mode": mode,
    }


def create_session_layout(condition: str) -> Tuple[str, Dict[str, str]]:
    """Create an isolated, append-only result tree; never delete another session's files."""
    session_id = uuid.uuid4().hex
    base_root = os.getenv("RESULTS_ROOT", "static/results")
    session_root = os.path.join(base_root, condition, session_id)
    paths = {
        "root": session_root,
        "base": os.path.join(session_root, "base"),
        "audit": os.path.join(session_root, "audit"),
        "final": os.path.join(session_root, "final_images"),
        "upscale_input": os.path.join(session_root, "upscale_input"),
        "upscale_output": os.path.join(session_root, "upscale_output"),
        "seam_initial": os.path.join(session_root, "seam_initial"),
        "seam_retry": os.path.join(session_root, "seam_retry"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return session_id, paths


def public_static_url(path: str) -> str:
    normalized = path.replace("\\", "/")
    marker = "/static/"
    if marker in f"/{normalized}":
        normalized = "static/" + f"/{normalized}".split(marker, 1)[1]
    return "/" + normalized.lstrip("/")


def new_seed(previous_seed: int | None = None, force_new_segment: bool = False) -> int:
    if force_new_segment and previous_seed is not None:
        return (int(previous_seed) + 1_000_003) % (2 ** 32)
    return secrets.randbelow(2 ** 32)


def _number(metrics: Dict[str, Any], key: str):
    try:
        return float(metrics[key])
    except (KeyError, TypeError, ValueError):
        return None


def candidate_rank(decision: str, metrics: Dict[str, Any], scene_data: Dict[str, Any]) -> Tuple[int, int, float, float]:
    """Rank candidates with PASS first, then available deterministic quality signals."""
    values = [value for value in (_number(metrics, "ds_score"), _number(metrics, "md_score")) if value is not None]
    quality = sum(values) / len(values) if values else -1.0
    target = _number(scene_data, "target_kelvin")
    actual = _number(metrics, "estimated_kelvin")
    kelvin_fit = -abs(actual - target) if actual is not None and target is not None else float("-inf")
    return (1 if str(decision).upper() == "PASS" else 0, len(values), quality, kelvin_fit)


def remember_candidate(data: Dict[str, Any], audit_report: Dict[str, Any], metrics: Dict[str, Any], round_index: int) -> bool:
    rank = candidate_rank(audit_report.get("decision", "FAIL"), metrics, data.get("scene_data") or {})
    current_best = data.get("best_candidate")
    if current_best is not None and tuple(current_best.get("rank", ())) >= rank:
        return False
    data["best_candidate"] = {
        "rank": rank,
        "round": round_index,
        "image_path": data.get("image_path"),
        "base_image_path": data.get("base_image_path"),
        "scene_data": deepcopy(data.get("scene_data") or {}),
        "reference_image_path": data.get("reference_image_path"),
        "reference_filename": data.get("reference_filename"),
        "audit_report": deepcopy(audit_report),
        "metrics": deepcopy(metrics),
        "is_passed": str(audit_report.get("decision", "")).upper() == "PASS",
        "seed": data.get("seed"),
    }
    return True


def select_best_candidate(data: Dict[str, Any]) -> Dict[str, Any]:
    best = data.get("best_candidate")
    if not best:
        return data
    for key in (
        "image_path", "base_image_path", "scene_data", "reference_image_path",
        "reference_filename", "audit_report", "metrics", "is_passed", "seed",
    ):
        data[key] = deepcopy(best.get(key))
    data["selected_candidate_round"] = best.get("round")
    return data


def _critique_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", (text or "").lower()))


def critique_similarity(left: str, right: str) -> float:
    a, b = _critique_tokens(left), _critique_tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def record_critique(data: Dict[str, Any], critique: str, threshold: float = 0.85) -> bool:
    history = data.setdefault("critique_history", [])
    previous = history[-1]["text"] if history else ""
    similarity = critique_similarity(previous, critique) if history else 0.0
    fingerprint = hashlib.sha256((critique or "").strip().lower().encode("utf-8")).hexdigest()[:16]
    history.append({"text": critique or "", "fingerprint": fingerprint, "similarity_to_previous": similarity})
    data["stagnation_count"] = data.get("stagnation_count", 0) + 1 if history[:-1] and similarity >= threshold else 0
    return data["stagnation_count"] >= 1


def append_session_record(final_result: Dict[str, Any], session_logs: Dict[str, Any]) -> str:
    path = Path(os.getenv("SESSION_LOG_JSONL", "static/results/session_logs.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"recorded_at": time.time(), "result": final_result, "logs": session_logs}
    with _JSONL_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return str(path)
