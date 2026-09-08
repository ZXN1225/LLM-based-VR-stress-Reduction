import json
import cv2
import base64
import numpy as np
from openai import OpenAI
from litellm import completion


class AuditingSkills:
    """Encapsulated Professional Skills for the Therapist Agent"""

    @staticmethod
    def encode_image(image_path):
        """Skill 1: Visual Perception - Reads the physical file for LLM vision"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def get_clinical_guidelines():
        """
        Skill 2: Knowledge Grounding -
        Provides the theoretical framework (The 'Knowledge Graph' equivalent)
        """
        return {
            "SRT_Stress_Reduction_Mechanism": {
                "Theory": "Evolutionary-based rapid recovery of the Autonomic Nervous System. When we are in a natural, non-threatening environment, our bodies quickly generate positive emotions to help us relax.",
                "Physiological_Impact": {
                    "Parasympathetic_Activation": "Low-arousal warm lighting (2700K-4500K) acts as a non-threatening signal, reducing cortisol levels.",
                    "Prospect_Refuge_Balance": "Visual 'Prospect' (open view) provides safety through information, while 'Refuge' (protected enclosure) reduces defensive vigilance.",
                    "Aversion_Triggers": "Sharp geometric edges or high-contrast artificial structures can trigger mild 'Fight-or-Flight' responses, hindering relaxation."
                }
            },
            "ART_Attention_Restoration_Mechanism": {
                "Theory": "Recovery of 'Directed Attention' through effortless fascination. For the brain to recover, an environment needs to have the following four characteristics:Being Away, Extent, Fascination, Compatibility",
                "Cognitive_Indicators": {
                    "Soft_Fascination": "Non-kinetic or rhythmic natural stimuli (leaves, clouds) that boost Normalized Alpha brainwaves (β=0.487 for greenery). Must be non-aggressive.",
                    "Extent_and_Being_Away": "The environment has enough scope and depth to immerse one in it to allow cognitive detachment from daily stressors.",
                    "Complexity_Management": "Optimal Fractal Dimension (D=1.3-1.5) induces Alpha-wave brain activity. Excessive complexity (Visual Chaos) leads to sensory overload.",
                    "Compatibility": "Environmental characteristics align with individual goals and needs. Match the scene complexity to the user's fatigue level. "
                }
            },
            "Neuro_Aesthetics_Parameters": {
                "Circadian_Entrainment": "High blue-light ratios (>6000K) suppress melatonin; appropriate for morning energy but detrimental for evening stress relief.",
                "Sensory_Overload_Prevention": "Excessive saturation or chaotic spatial frequencies increase cognitive load, violating the 'Restorative' intent."
            },
            "Urban_Density_Mitigation": {
                "Theory": "Pathways model linking streetscapes to stress via Perceived Oppressiveness (PO).",
                "Key_Interventions": {
                    "Sky_Ratio": "Maintain high sky-view factor to reduce PO and increase sense of freedom.",
                    "Visual_Buffer": "Use Tree Canopies to shield artificial facades/buildings to lower psychological pressure.",
                    "Detrimental_Elements": "Strictly limit vehicles and aggressive billboards which explain 50.2% of mental stress."
                }
            },
            "Clinical_Audit_Strategies": {
                "Anxiety_Intervention": "Focus: Limit kinetic stimuli, reduce color contrast, and emphasize 'Refuge' elements.",
                "Burnout_Recovery": "Focus: Enhance 'Being Away' and provide expansive 'Prospects' to rebuild spatial cognition."
            }
        }

    @staticmethod
    def calculate_alignment(client, prompt, clinical_insight):
        """Skill 3: Semantic Alignment - Quantifies intent matching"""
        res = client.embeddings.create(input=[prompt, clinical_insight], model="text-embedding-3-small")
        v1, v2 = res.data[0].embedding, res.data[1].embedding
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    @staticmethod
    def calculate_visual_metrics(image_path):
        """Skill 4: Calculate saturation and Contrast Ratio"""
        img = cv2.imread(image_path)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        avg_saturation = np.mean(hsv_img[:, :, 1]) / 255.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
        rms_contrast = np.std(gray)

        return {
            "avg_saturation": round(float(avg_saturation), 2),
            "rms_contrast": round(float(rms_contrast), 2)
        }



class AuditingAgent:
    _cached_system_prompt = None

    def __init__(self, api_key, openai_api, model_name="gpt-5.4"):
        self.model = model_name
        self.client = OpenAI(api_key=openai_api)
        self.api_key = api_key
        self.skills = AuditingSkills()

    @classmethod
    def _static_system_prompt(cls):
        """Stable prefix so repeated scene audits can use provider prompt caching."""
        if cls._cached_system_prompt is None:
            guidelines = AuditingSkills.get_clinical_guidelines()
            cls._cached_system_prompt = f"""
You are a Senior VR Restorative Environment Auditor specializing in stress reduction and Environmental Psychology.
Audit the generated VR scene using visual inspection and quantitative data. If it fails, provide concise,
actionable instructions to the scene refiner.

[CLINICAL KNOWLEDGE BASE]
{json.dumps(guidelines, indent=2, sort_keys=True)}

[AUDIT PROTOCOL]
1. SAFETY AND IMMERSION: reject catastrophic seam discontinuity, abnormal realism score, or visible glitches
   likely to cause nausea or break immersion.
2. PERSONALIZATION: match the supplied psychological state, preferences, sensitivities, and avoid-elements.
3. SRT/ART: explain why the scene works or fails for this specific state.
4. COMPLEXITY: interpret complexity and fractal metrics with the image; do not reject a natural-looking forest
   solely because a metric is high.
5. SOFT PASS: pass an immersive, restorative scene without major defects even when metrics slightly miss targets.
6. Quantitative metrics are supporting evidence, not uncalibrated absolute clinical thresholds.

[OUTPUT FORMAT]
Return exactly one valid JSON object:
{{
  "decision": "PASS" or "FAIL",
  "clinical_critique": "concise explanation",
  "refinement_suggestion": "specific corrective instruction; empty when no correction is needed"
}}
""".strip()
        return cls._cached_system_prompt

    def audit_scene(self, image_path, scene_data, physical_metrics, user_input, clinical_insight, psych_state=None):
        """
        Multimodal Clinical Audit:
        Integrates [Visual Image] + [Physical Metrics] + [Expert Knowledge]
        """
        # 1. READ IMAGE
        base64_image = self.skills.encode_image(image_path)

        # 2. RETRIEVE KNOWLEDGE & CALCULATE ALIGNMENT
        psych_state = psych_state or clinical_insight.get("psych_state", {}) or {}
        target_strategy = clinical_insight.get('search_query', user_input)
        alignment_score = self.skills.calculate_alignment(
            self.client, scene_data['image_prompt'], target_strategy
        )
        visual_metrics = self.skills.calculate_visual_metrics(image_path)
        saturation = visual_metrics.get("avg_saturation", 0)
        contrast_ratio = visual_metrics.get("rms_contrast", 0)

        # 3. Keep the system prefix static; all per-scene content belongs at the end/user turn.
        system_prompt = self._static_system_prompt()

        user_content = [
            {
                "type": "text",
                "text": (
                    f"Evaluate this scene for the user's original need: {user_input}\n"
                    f"Scene data: {json.dumps(scene_data, ensure_ascii=False)}\n"
                    f"Dialog-derived psych_state: {json.dumps(psych_state, ensure_ascii=False)}\n"
                    f"Physical metrics: {json.dumps(physical_metrics)}\n"
                    f"Semantic alignment score: {alignment_score:.4f}\n"
                    f"Visual harmony metrics: {json.dumps({'avg_saturation': saturation, 'rms_contrast': contrast_ratio})}"
                )
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                api_key=self.api_key,
                response_format={"type": "json_object"},
                num_retries=3,
                timeout=120.0
            )
            res_content = response.choices[0].message.content
            report = json.loads(res_content)
            if not isinstance(report, dict) or str(report.get("decision", "")).upper() not in {"PASS", "FAIL"}:
                raise ValueError("Audit response must contain decision PASS or FAIL")
            report["decision"] = str(report["decision"]).upper()
            report.setdefault("clinical_critique", "")
            report.setdefault("refinement_suggestion", "")
            return report

        except Exception as e:
            print(f"⚠️ Therapist Audit Failed: {e}")
            return {"decision": "FAIL", "clinical_critique": f"Audit Error: {str(e)}",
                    "refinement_suggestion": "Retry with original prompt"}
