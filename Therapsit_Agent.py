import json
import cv2
import base64
import numpy as np
from openai import OpenAI


class TherapistSkills:
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
                    "Soft_Fascination": "Stimuli that capture attention without cognitive effort (e.g., rustling leaves, water ripples). Must be non-aggressive.",
                    "Extent_and_Being_Away": "The environment must provide a sense of 'immersion in another world' to allow cognitive detachment from daily stressors.",
                    "Complexity_Management": "Optimal Fractal Dimension (D=1.3-1.5) induces Alpha-wave brain activity. Excessive complexity (Visual Chaos) leads to sensory overload."
                }
            },
            "Neuro_Aesthetics_Parameters": {
                "Circadian_Entrainment": "High blue-light ratios (>6000K) suppress melatonin; appropriate for morning energy but detrimental for evening stress relief.",
                "Sensory_Overload_Prevention": "Excessive saturation or chaotic spatial frequencies increase cognitive load, violating the 'Restorative' intent."
            },
            "Clinical_Audit_Strategies": {
                "Anxiety_Intervention": "Focus: Limit kinetic stimuli, reduce color contrast, and emphasize 'Refuge' elements.",
                "Burnout_Recovery": "Focus: Enhance 'Being Away' and provide expansive 'Prospects' to rebuild spatial cognition."
            }
        }

    @staticmethod
    def calculate_alignment(client, prompt, user_input):
        """Skill 3: Semantic Alignment - Quantifies intent matching"""
        res = client.embeddings.create(input=[prompt, user_input], model="text-embedding-3-small")
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



class TherapistAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.skills = TherapistSkills()

    def audit_scene(self, image_path, scene_data, physical_metrics, user_input):
        """
        Multimodal Clinical Audit:
        Integrates [Visual Image] + [Physical Metrics] + [Expert Knowledge]
        """
        # 1. READ IMAGE
        base64_image = self.skills.encode_image(image_path)

        # 2. RETRIEVE KNOWLEDGE & CALCULATE ALIGNMENT
        guidelines = self.skills.get_clinical_guidelines()
        alignment_score = self.skills.calculate_alignment(
            self.client, scene_data['image_prompt'], user_input
        )
        saturation, contrast_ratio = self.skills.calculate_visual_metrics(image_path)

        # 3. CONSTRUCT MULTIMODAL EXPERT PROMPT
        system_prompt = f"""
        You are a Senior VR Clinical Therapist specializing in stress reduction. 
        Your task is to audit the generated VR environment(scene) using both visual inspection and quantitative data. If the scene is not passed,
        you need provide refining suggestion to refiner to generate images that fits more to user input and stress reduction therapy.

        [CLINICAL KNOWLEDGE BASE]:
        {json.dumps(guidelines, indent=2)}

        [QUANTITATIVE DATA]:
        - Physical Metrics : {json.dumps(physical_metrics)}
        - Semantic Alignment Score(prompt to user input): {alignment_score:.2f}
        - Visual Harmony Metrics: {json.dumps({'avg_saturation': saturation, 'rms_contrast': contrast_ratio})}

        [AUDIT PROTOCOL]:
        1. SAFETY & IMMERSION: Reject if DS-Score or FAED is too low or abnormal. Visual glitches cause nausea and break the 'Being Away' state. Visual Harmony Metrics should not be too high which cause Excessive visual stimulation.
        2. SRT/ART ANALYSIS: Use the 'Mechanism' in the Knowledge Base to explain WHY the scene works or fails. 
           Example: Don't just say 'It's bright'; say 'The high luminance and cool color temperature may inhibit melatonin and fail to activate the parasympathetic nervous system'.
        3. FRACTAL AUDIT: Evaluate if the 'Complexity' metric suggests restorative natural patterns or stressful visual chaos.
        4. SOFT PASS LOGIC: If the visual content is restorative but physical metrics (like Kelvin and others) are slightly off, output "PASS" as well. 
        5. Remember you are a professional VR Clinical Therapist. You should utilize or base on your professional knowledge base and how you think of the scene, physical metrics are for reference only. Do not rely on them too much and be too strict.
        
        [OUTPUT FORMAT]:
        You MUST return the final audit report in a valid **JSON** format. 
        The **JSON** object must follow this structure:
        {{
            "decision": "PASS" or "FAIL"
            "clinical_critique": "Detailed medical explanation of why it passed or failed",
            "refinement_suggestion": "Specific instructions for the RAG Agent to improve the prompt/parameters",
        }}
        """
        for attempt in range(6):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Evaluate this scene for the user's need: {user_input}"},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
                    ],
                    response_format={"type": "json_object"},
                    timeout = 60.0
                )
                res_content = response.choices[0].message.content
                return json.loads(res_content)

            except Exception as e:
                print(f"⚠️ Therapist Audit Attempt {attempt + 1} Failed: {e}")
                if attempt == 5:
                    return {
                        "decision": "FAIL",
                        "clinical_critique": "Audit failed due to API timeout or error.",
                        "refinement_suggestion": "Retry the generation process."
                    }