import json
from copy import deepcopy
from typing import Any, Dict, Tuple
from litellm import completion
from PipelineGuardrails import detect_risk_flags


class DialogTherapistAgent:
    """
    Dialog-guided psychological state elicitation agent.

    It does NOT generate visual prompts directly. Its responsibility is to:
    1) conduct a friendly, short, multi-turn dialogue,
    2) update a structured psychological state,
    3) decide whether enough information has been collected,
    4) provide a final clinical summary for RAG-based intervention planning.
    """

    DEFAULT_STATE: Dict[str, Any] = {
        "history": [],
        "psych_state": {
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
        },
        "confidence": 0.0,
        "turn": 0,
        "finalized": False,
        "last_reply": "",
        "next_question": "",
    }

    def __init__(self, api_key: str, model_name: str = "gpt-5.4", max_turns: int = 5):
        self.api_key = api_key
        self.model = model_name
        self.max_turns = max_turns

    def new_state(self, initial_user_input: str = "") -> Dict[str, Any]:
        state = deepcopy(self.DEFAULT_STATE)
        if initial_user_input:
            state["history"].append({"role": "user", "content": initial_user_input})
            state["turn"] = 1
        return state

    @staticmethod
    def _safe_json_loads(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except Exception:
                    pass
        return {}

    @staticmethod
    def _merge_psych_state(old_state: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(old_state or {})
        for key, value in (update or {}).items():
            if value in [None, "", "unknown", [], {}]:
                continue
            if isinstance(value, list):
                existing = merged.get(key, [])
                if not isinstance(existing, list):
                    existing = [existing]
                merged[key] = list(dict.fromkeys(existing + value))
            else:
                merged[key] = value
        return merged

    def _system_prompt(self) -> str:
        return """
You are a friendly VR stress-reduction dialogue therapist and environmental psychology planner.

Your job is NOT to diagnose or provide medical treatment. Your job is to gently elicit enough information to personalize a restorative VR intervention.

Use the following theory-grounded reasoning silently:
- SRT: reduce threat, support parasympathetic recovery, balance prospect/refuge, avoid harsh contrast and threatening geometry.
- ART: support Being Away, Extent, Soft Fascination, and Compatibility; adjust complexity to fatigue level.
- SAD/low mood: usually needs safe brightness, warm light, open sky, and gentle activation.
- High anxiety: usually needs low stimulation, safety/refuge, low contrast, stable horizon, and predictable natural elements.
- Cognitive fatigue/burnout: usually needs openness, depth, soft fascination, and low-to-moderate complexity.

Conversation style:
- Warm, concise, and supportive.
- Ask only ONE clear follow-up question at a time.
- Do not sound like a questionnaire.
- Avoid medical diagnosis language.
- If the user mentions severe self-harm intent or immediate danger, set risk_flags and recommend seeking immediate human help.

You must update the psychological state each turn.
You should finish when confidence >= 0.80 or turn >= max_turns, unless there is a major missing preference.

Return ONLY valid JSON:
{
  "reply": "friendly response shown to the user",
  "state_update": {
    "stress_type": "cognitive_fatigue | high_arousal_anxiety | low_mood_sad | burnout | loneliness | mixed | unknown",
    "primary_emotion": "...",
    "arousal": 0.0,
    "fatigue": 0.0,
    "valence": -1.0,
    "stress_intensity": 0.0,
    "needs": ["..."],
    "triggers": ["..."],
    "environment_preference": "...",
    "lighting_preference": "...",
    "sensory_sensitivity": "low | moderate | high | unknown",
    "avoid_elements": ["..."],
    "target_srt_art_mechanisms": ["prospect", "refuge", "soft_fascination", "being_away", "extent", "compatibility"],
    "risk_flags": [],
    "session_goal": "..."
  },
  "confidence": 0.0,
  "should_finish": false,
  "next_question": "one short follow-up question or empty string"
}
"""

    def step(self, user_message: str, state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not isinstance(state, dict) or not state:
            state = self.new_state()

        state = deepcopy(state)
        state.setdefault("history", [])
        state.setdefault("psych_state", deepcopy(self.DEFAULT_STATE["psych_state"]))
        state["turn"] = int(state.get("turn", 0)) + 1
        state["history"].append({"role": "user", "content": user_message})

        compact_history = state["history"][-10:]

        user_prompt = {
            "conversation_history": compact_history,
            "current_psych_state": state.get("psych_state", {}),
            "current_turn": state.get("turn", 1),
            "max_turns": self.max_turns,
            "instruction": "Continue the dialogue and update the psychological state."
        }

        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)}
                ],
                api_key=self.api_key,
                response_format={"type": "json_object"},
                num_retries=3,
                timeout=90.0,
            )
            content = response.choices[0].message.content
            out = self._safe_json_loads(content)
        except Exception as e:
            out = {
                "reply": "我理解了。为了更好地为你生成合适的VR放松环境，你现在更需要安静、安全感，还是更需要明亮和一点点能量？",
                "state_update": {},
                "confidence": state.get("confidence", 0.3),
                "should_finish": False,
                "next_question": "你现在更需要安静、安全感，还是更需要明亮和一点点能量？",
                "error": str(e),
            }

        state["psych_state"] = self._merge_psych_state(
            state.get("psych_state", {}),
            out.get("state_update", {})
        )
        state["psych_state"]["risk_flags"] = detect_risk_flags(
            user_message, state["psych_state"]
        )
        previous_confidence = float(state.get("confidence", 0.0) or 0.0)
        reported_confidence = float(out.get("confidence", previous_confidence) or 0.0)
        state["confidence"] = max(previous_confidence, min(1.0, reported_confidence))
        state["finalized"] = bool(out.get("should_finish", False)) or state["turn"] >= self.max_turns
        state["last_reply"] = out.get("reply", "")
        state["next_question"] = out.get("next_question", "")
        state["history"].append({"role": "assistant", "content": state["last_reply"]})

        return out, state

    def finalize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state = deepcopy(state or self.new_state())
        state["finalized"] = True
        return state

    def build_planning_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        psych = deepcopy((state or {}).get("psych_state", {}))
        history = (state or {}).get("history", [])
        user_utterances = [m.get("content", "") for m in history if m.get("role") == "user"]
        original_need = "\n".join(user_utterances[-6:])

        stress_type = psych.get("stress_type", "mixed")
        needs = psych.get("needs", [])
        mechanisms = psych.get("target_srt_art_mechanisms", [])
        env_pref = psych.get("environment_preference", "restorative natural environment")
        light_pref = psych.get("lighting_preference", "warm soft light")

        summary = (
            f"Dialog-derived state: stress_type={stress_type}; "
            f"primary_emotion={psych.get('primary_emotion', 'unknown')}; "
            f"arousal={psych.get('arousal', 0.5)}; fatigue={psych.get('fatigue', 0.5)}; "
            f"valence={psych.get('valence', -0.2)}; needs={needs}; "
            f"preferred_environment={env_pref}; lighting={light_pref}; "
            f"target_SRT_ART_mechanisms={mechanisms}; "
            f"avoid_elements={psych.get('avoid_elements', [])}; "
            f"session_goal={psych.get('session_goal', 'stress relief')}."
        )

        return {
            "original_user_need": original_need,
            "psych_state": psych,
            "dialog_confidence": (state or {}).get("confidence", 0.0),
            "dialog_summary": summary,
            "search_query": self._build_search_query(psych),
        }

    @staticmethod
    def _build_search_query(psych: Dict[str, Any]) -> str:
        stress = psych.get("stress_type", "stress relief")
        env = psych.get("environment_preference", "restorative nature")
        light = psych.get("lighting_preference", "warm soft light")
        mechanisms = ", ".join(psych.get("target_srt_art_mechanisms", []) or [])
        needs = ", ".join(psych.get("needs", []) or [])
        avoid = ", ".join(psych.get("avoid_elements", []) or [])
        return (
            f"{stress}, {env}, {light}, needs: {needs}, "
            f"SRT/ART: {mechanisms}, avoid: {avoid}"
        )
