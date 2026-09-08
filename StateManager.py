import json
from copy import deepcopy
from typing import Any, Dict, List, Tuple


class StateManager:
    """
    Lightweight state utility for the dialog-guided VR therapy UI.

    This class keeps UI state independent from the LLM agent. It deliberately
    does not call any model. It only:
      1) normalizes dialog_state,
      2) converts role-based history to Gradio chatbot pairs,
      3) decides whether the generation button can be enabled,
      4) formats psych_state and status for display.

    Expected dialog_state shape is compatible with DialogTherapistAgent:
    {
        "history": [{"role": "user"|"assistant", "content": "..."}],
        "psych_state": {...},
        "confidence": float,
        "turn": int,
        "finalized": bool,
        "last_reply": str,
        "next_question": str
    }
    """

    DEFAULT_PSYCH_STATE: Dict[str, Any] = {
        "stress_type": "unknown",
        "primary_emotion": "unknown",
        "arousal": 0.5,
        "fatigue": 0.5,
        "valence": -0.2,
        "stress_intensity": 0.5,
        "needs": [],
        "triggers": [],
        "environment_preference": "unknown",
        "lighting_preference": "unknown",
        "sensory_sensitivity": "unknown",
        "avoid_elements": [],
        "target_srt_art_mechanisms": [],
        "risk_flags": [],
        "session_goal": "stress relief",
    }

    @classmethod
    def new_state(cls) -> Dict[str, Any]:
        return {
            "history": [],
            "psych_state": deepcopy(cls.DEFAULT_PSYCH_STATE),
            "confidence": 0.0,
            "turn": 0,
            "finalized": False,
            "last_reply": "",
            "next_question": "",
        }

    @classmethod
    def ensure_state(cls, state: Dict[str, Any] | None) -> Dict[str, Any]:
        if not isinstance(state, dict) or not state:
            return cls.new_state()

        normalized = deepcopy(state)
        normalized.setdefault("history", [])
        normalized.setdefault("psych_state", deepcopy(cls.DEFAULT_PSYCH_STATE))
        normalized.setdefault("confidence", 0.0)
        normalized.setdefault("turn", 0)
        normalized.setdefault("finalized", False)
        normalized.setdefault("last_reply", "")
        normalized.setdefault("next_question", "")

        # Fill missing psych_state keys without overwriting useful values.
        psych = normalized.get("psych_state") or {}
        for k, v in cls.DEFAULT_PSYCH_STATE.items():
            psych.setdefault(k, deepcopy(v))
        normalized["psych_state"] = psych

        # Keep history entries safe and minimal.
        clean_history = []
        for item in normalized.get("history", []):
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content", "")
            if role in ["user", "assistant"] and isinstance(content, str):
                clean_history.append({"role": role, "content": content})
        normalized["history"] = clean_history
        return normalized

    @classmethod
    def chatbot_pairs(cls, state: Dict[str, Any] | None) -> List[Tuple[str, str]]:
        """
        Convert role-based history into the classic Gradio Chatbot format:
        [(user_message, assistant_message), ...]

        This avoids relying on gr.Chatbot(type="messages"), which is not
        available in some Gradio versions.
        """
        state = cls.ensure_state(state)
        pairs: List[Tuple[str, str]] = []
        pending_user: str | None = None

        for msg in state.get("history", []):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                if pending_user is not None:
                    pairs.append((pending_user, ""))
                pending_user = content
            elif role == "assistant":
                if pending_user is None:
                    pairs.append(("", content))
                else:
                    pairs.append((pending_user, content))
                    pending_user = None

        if pending_user is not None:
            pairs.append((pending_user, ""))
        return pairs


    @classmethod
    def chatbot_messages(cls, state: Dict[str, Any] | None) -> List[Dict[str, str]]:
        """
        Convert role-based history into Gradio's messages format:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Your installed Gradio expects the messages format by default, even though
        Chatbot(type="messages") is not accepted in the constructor.
        Therefore we keep the constructor simple and return message dictionaries.
        """
        state = cls.ensure_state(state)
        messages: List[Dict[str, str]] = []
        for msg in state.get("history", []):
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ["user", "assistant"] and isinstance(content, str):
                messages.append({"role": role, "content": content})
        return messages

    @classmethod
    def completeness(cls, state: Dict[str, Any] | None) -> Dict[str, bool]:
        """Deterministic readiness checklist; model confidence is display-only."""
        state = cls.ensure_state(state)
        psych = state.get("psych_state", {}) or {}

        def known(value):
            return bool(value) and str(value).strip().lower() not in {"unknown", "none", "n/a"}

        return {
            "stress_type": known(psych.get("stress_type")),
            "environment_preference": known(psych.get("environment_preference")),
            "lighting_preference": known(psych.get("lighting_preference")),
            "target_srt_art_mechanisms": bool(psych.get("target_srt_art_mechanisms")),
            "avoid_elements": bool(psych.get("avoid_elements")),
        }

    @classmethod
    def has_risk(cls, state: Dict[str, Any] | None) -> bool:
        state = cls.ensure_state(state)
        return bool((state.get("psych_state", {}) or {}).get("risk_flags"))

    @classmethod
    def can_generate(cls, state: Dict[str, Any] | None, threshold: float = 0.80) -> bool:
        state = cls.ensure_state(state)
        if cls.has_risk(state):
            return False
        complete = all(cls.completeness(state).values())
        return bool(state.get("finalized", False)) or complete

    @classmethod
    def finalize(cls, state: Dict[str, Any] | None) -> Dict[str, Any]:
        state = cls.ensure_state(state)
        state["finalized"] = True
        return state

    @classmethod
    def append_assistant(cls, state: Dict[str, Any] | None, content: str) -> Dict[str, Any]:
        state = cls.ensure_state(state)
        if content:
            state["history"].append({"role": "assistant", "content": content})
            state["last_reply"] = content
        return state

    @classmethod
    def psych_state_json(cls, state: Dict[str, Any] | None) -> str:
        state = cls.ensure_state(state)
        return json.dumps(state.get("psych_state", {}), indent=2, ensure_ascii=False)

    @classmethod
    def full_state_json(cls, state: Dict[str, Any] | None) -> str:
        state = cls.ensure_state(state)
        return json.dumps(state, indent=2, ensure_ascii=False)

    @classmethod
    def status_markdown(cls, state: Dict[str, Any] | None) -> str:
        state = cls.ensure_state(state)
        psych = state.get("psych_state", {}) or {}
        confidence = float(state.get("confidence", 0.0) or 0.0)
        finalized = bool(state.get("finalized", False))
        turn = int(state.get("turn", 0) or 0)

        stress_type = psych.get("stress_type", "unknown")
        emotion = psych.get("primary_emotion", "unknown")
        arousal = psych.get("arousal", 0.5)
        fatigue = psych.get("fatigue", 0.5)
        env = psych.get("environment_preference", "unknown")
        light = psych.get("lighting_preference", "unknown")
        mechanisms = psych.get("target_srt_art_mechanisms", []) or []
        needs = psych.get("needs", []) or []
        risk_flags = psych.get("risk_flags", []) or []

        if risk_flags:
            ready_text = "⛔ Generation blocked by safety gate"
        else:
            ready_text = "✅ Ready to generate" if cls.can_generate(state) else "⏳ Continue dialog or click Use current state"
        risk_text = "\n\n⚠️ **Risk flags detected:** " + ", ".join(risk_flags) if risk_flags else ""

        return (
            f"### Dialog State\n"
            f"- **Status:** {ready_text}\n"
            f"- **Turns:** {turn}\n"
            f"- **Confidence:** {confidence:.2f}\n"
            f"- **Finalized:** {finalized}\n\n"
            f"### Psychological Estimate\n"
            f"- **Stress type:** `{stress_type}`\n"
            f"- **Primary emotion:** `{emotion}`\n"
            f"- **Arousal:** `{arousal}`\n"
            f"- **Fatigue:** `{fatigue}`\n"
            f"- **Needs:** {', '.join(map(str, needs)) if needs else 'unknown'}\n"
            f"- **Environment:** {env}\n"
            f"- **Light:** {light}\n"
            f"- **SRT/ART targets:** {', '.join(map(str, mechanisms)) if mechanisms else 'unknown'}"
            f"{risk_text}"
        )
