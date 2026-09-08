import json
from litellm import completion

class SinglePromptPsychStateExtractor:
    """
    One-shot structured psychological-state extractor for the passive condition.

    The extractor receives only the participant's free-text prompt and returns the
    same psych_state schema used by the dialog-guided condition. It does not ask
    follow-up questions and does not generate scene prompts directly.
    """

    DEFAULT_STATE = {
        "stress_type": "mixed",
        "primary_emotion": "unknown",
        "arousal": 0.5,
        "fatigue": 0.5,
        "valence": -0.2,
        "stress_intensity": 0.5,
        "needs": ["stress relief", "mental recovery"],
        "triggers": [],
        "environment_preference": "restorative natural environment",
        "lighting_preference": "warm soft natural light",
        "sensory_sensitivity": "moderate",
        "avoid_elements": ["crowds", "vehicles", "harsh contrast"],
        "target_srt_art_mechanisms": [
            "being_away", "soft_fascination", "compatibility"
        ],
        "risk_flags": [],
        "session_goal": "stress relief",
    }

    ALLOWED_STRESS_TYPES = {
        "cognitive_fatigue",
        "high_arousal_anxiety",
        "low_mood_sad",
        "burnout",
        "loneliness",
        "mixed",
        "unknown",
    }
    ALLOWED_SENSORY_LEVELS = {"low", "moderate", "high", "unknown"}
    ALLOWED_MECHANISMS = {
        "prospect", "refuge", "soft_fascination",
        "being_away", "extent", "compatibility",
    }

    def __init__(self, api_key: str, model_name: str = "gpt-5.4"):
        self.api_key = api_key
        self.model = model_name

    @staticmethod
    def _safe_json_loads(text: str) -> dict:
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
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            items = value
        else:
            items = str(value).replace(";", ",").split(",")
        clean = []
        for item in items:
            text = str(item).strip()
            if text and text.lower() not in {"unknown", "none", "n/a"}:
                clean.append(text)
        return list(dict.fromkeys(clean))

    @staticmethod
    def _bounded_float(value, default, low, high):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float(default)
        return max(low, min(high, number))

    @classmethod
    def _defaults_for_stress_type(cls, stress_type: str) -> dict:
        if stress_type == "high_arousal_anxiety":
            return {
                "needs": ["safety", "calm", "sense of control"],
                "environment_preference": "quiet predictable natural environment with open prospect and gentle refuge",
                "lighting_preference": "warm soft low-contrast natural light",
                "avoid_elements": ["crowds", "chaotic activity", "harsh contrast", "threatening enclosed spaces"],
                "target_srt_art_mechanisms": ["refuge", "prospect", "soft_fascination", "compatibility"],
            }
        if stress_type in {"cognitive_fatigue", "burnout"}:
            return {
                "needs": ["mental recovery", "detachment", "effortless attention"],
                "environment_preference": "spacious natural landscape with depth and gentle natural detail",
                "lighting_preference": "warm-to-neutral soft natural brightness",
                "avoid_elements": ["visual clutter", "crowds", "harsh artificial structures"],
                "target_srt_art_mechanisms": ["being_away", "extent", "soft_fascination", "compatibility"],
            }
        if stress_type == "low_mood_sad":
            return {
                "needs": ["gentle activation", "emotional warmth", "hopefulness"],
                "environment_preference": "bright open natural environment with sky and welcoming depth",
                "lighting_preference": "warm clear natural brightness",
                "avoid_elements": ["dark enclosed spaces", "gloom", "harsh contrast"],
                "target_srt_art_mechanisms": ["prospect", "extent", "soft_fascination", "compatibility"],
            }
        if stress_type == "loneliness":
            return {
                "needs": ["comfort", "connection", "emotional safety"],
                "environment_preference": "welcoming calm natural environment with a sheltered resting place",
                "lighting_preference": "warm gentle natural light",
                "avoid_elements": ["bleak emptiness", "threatening isolation", "harsh contrast"],
                "target_srt_art_mechanisms": ["refuge", "soft_fascination", "being_away", "compatibility"],
            }
        return {
            "needs": ["stress relief", "mental recovery"],
            "environment_preference": "restorative natural environment",
            "lighting_preference": "warm soft natural light",
            "avoid_elements": ["crowds", "vehicles", "harsh contrast"],
            "target_srt_art_mechanisms": ["being_away", "soft_fascination", "compatibility"],
        }

    @classmethod
    def normalize(cls, raw_state: dict | None) -> dict:
        raw = raw_state if isinstance(raw_state, dict) else {}
        stress_type = str(raw.get("stress_type", "mixed") or "mixed").strip().lower()
        if stress_type not in cls.ALLOWED_STRESS_TYPES:
            stress_type = "mixed"

        defaults = cls._defaults_for_stress_type(stress_type)
        state = dict(cls.DEFAULT_STATE)
        state.update(defaults)

        state["stress_type"] = stress_type
        state["primary_emotion"] = str(raw.get("primary_emotion") or state["primary_emotion"]).strip()
        state["arousal"] = cls._bounded_float(raw.get("arousal"), state["arousal"], 0.0, 1.0)
        state["fatigue"] = cls._bounded_float(raw.get("fatigue"), state["fatigue"], 0.0, 1.0)
        state["valence"] = cls._bounded_float(raw.get("valence"), state["valence"], -1.0, 1.0)
        state["stress_intensity"] = cls._bounded_float(
            raw.get("stress_intensity"), state["stress_intensity"], 0.0, 1.0
        )

        for key in ["needs", "triggers", "avoid_elements", "risk_flags"]:
            parsed = cls._as_list(raw.get(key))
            if parsed or key in {"triggers", "risk_flags"}:
                state[key] = parsed

        mechanisms = [
            str(x).strip().lower()
            for x in cls._as_list(raw.get("target_srt_art_mechanisms"))
        ]
        mechanisms = [x for x in mechanisms if x in cls.ALLOWED_MECHANISMS]
        if mechanisms:
            state["target_srt_art_mechanisms"] = list(dict.fromkeys(mechanisms))

        for key in [
            "environment_preference", "lighting_preference", "session_goal"
        ]:
            value = raw.get(key)
            if value not in [None, "", "unknown"]:
                state[key] = str(value).strip()

        sensory = str(raw.get("sensory_sensitivity", state["sensory_sensitivity"]) or "moderate").strip().lower()
        state["sensory_sensitivity"] = sensory if sensory in cls.ALLOWED_SENSORY_LEVELS else "moderate"
        return state

    def _system_prompt(self) -> str:
        return """
You are a one-shot psychological-state extractor for a non-clinical, personalized
restorative VR system. Analyze only the user's single free-text prompt. Do not ask
follow-up questions and do not generate panorama or music prompts.

Infer a complete structured state using SRT and ART, while remaining conservative:
- Do not diagnose a medical condition.
- Do not invent specific life events, preferences, or triggers that are absent.
- Numeric estimates are allowed when the wording supports them; otherwise use moderate defaults.
- If a visual preference is absent, choose a safe theory-grounded restorative default.
- Convert user dislikes, sensitivities, and threatening/stimulating content into avoid_elements.
- Select only these mechanisms: prospect, refuge, soft_fascination, being_away, extent, compatibility.
- If the prompt indicates immediate danger or self-harm intent, record concise risk_flags.

Stress-type labels:
- cognitive_fatigue
- high_arousal_anxiety
- low_mood_sad
- burnout
- loneliness
- mixed
- unknown

Return ONLY valid JSON with this exact top-level shape:
{
  "psych_state": {
    "stress_type": "...",
    "primary_emotion": "...",
    "arousal": 0.0,
    "fatigue": 0.0,
    "valence": 0.0,
    "stress_intensity": 0.0,
    "needs": ["..."],
    "triggers": ["..."],
    "environment_preference": "...",
    "lighting_preference": "...",
    "sensory_sensitivity": "low | moderate | high | unknown",
    "avoid_elements": ["..."],
    "target_srt_art_mechanisms": ["..."],
    "risk_flags": ["..."],
    "session_goal": "..."
  },
  "confidence": 0.0,
  "evidence_summary": "Briefly state which parts were explicit and which were inferred."
}
"""

    def extract(self, user_input: str):
        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": f"User prompt:\n{user_input}"},
                ],
                api_key=self.api_key,
                response_format={"type": "json_object"},
                num_retries=3,
                timeout=90.0,
            )
            parsed = self._safe_json_loads(response.choices[0].message.content)
            raw_state = parsed.get("psych_state", parsed)
            psych_state = self.normalize(raw_state)
            confidence = self._bounded_float(parsed.get("confidence"), 0.65, 0.0, 1.0)
            metadata = {
                "source": "single_prompt_llm_extraction",
                "confidence": confidence,
                "evidence_summary": str(parsed.get("evidence_summary", "")).strip(),
                "error": None,
            }
            return psych_state, metadata
        except Exception as exc:
            psych_state = self.normalize({})
            metadata = {
                "source": "fallback_default_state",
                "confidence": 0.0,
                "evidence_summary": "Automatic extraction failed; safe generic defaults were used.",
                "error": str(exc),
            }
            return psych_state, metadata

def build_passive_context(user_input: str, psych_state: dict, extraction_confidence: float=1.0):
    psych_state = psych_state or {}
    stress_type = psych_state.get('stress_type', 'mixed')
    needs = psych_state.get('needs', [])
    mechanisms = psych_state.get('target_srt_art_mechanisms', [])
    env_pref = psych_state.get('environment_preference', 'restorative natural environment')
    light_pref = psych_state.get('lighting_preference', 'warm soft light')
    avoid = psych_state.get('avoid_elements', [])
    dialog_summary = f"Single-prompt extracted state: stress_type={stress_type}; primary_emotion={psych_state.get('primary_emotion', 'unknown')}; arousal={psych_state.get('arousal', 0.5)}; fatigue={psych_state.get('fatigue', 0.5)}; valence={psych_state.get('valence', -0.2)}; stress_intensity={psych_state.get('stress_intensity', 0.5)}; needs={needs}; triggers={psych_state.get('triggers', [])}; preferred_environment={env_pref}; lighting={light_pref}; target_SRT_ART_mechanisms={mechanisms}; avoid_elements={avoid}; session_goal={psych_state.get('session_goal', 'stress relief')}."
    search_query = f"{stress_type}, {env_pref}, {light_pref}, needs: {(', '.join(needs) if isinstance(needs, list) else needs)}, SRT/ART: {(', '.join(mechanisms) if isinstance(mechanisms, list) else mechanisms)}, avoid: {(', '.join(avoid) if isinstance(avoid, list) else avoid)}"
    return {'original_user_need': user_input, 'psych_state': psych_state, 'dialog_confidence': float(extraction_confidence), 'state_extraction_confidence': float(extraction_confidence), 'dialog_summary': dialog_summary, 'search_query': search_query}
