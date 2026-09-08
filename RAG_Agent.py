import os
import json
import re
from copy import deepcopy
import chromadb
from chromadb.utils import embedding_functions
from litellm import completion
from pydantic import ValidationError
from PlanSchemas import InterventionPlanSpec, RefinementPlanSpec


class RAGAgent:
    def __init__(self, api_key, chroma_api, model_name="gpt-5.4", db_path="./PictureBase", picture_data_dir="./PictureData"):
        self.api_key = api_key
        self.model = model_name
        self.picture_data_dir = picture_data_dir
        self.emb_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=chroma_api,
            model_name="text-embedding-3-small"
        )
        self.db_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.db_client.get_collection(
            name="nature_environments",
            embedding_function=self.emb_fn
        )

    def _validated_completion(self, messages, schema):
        """Validate model JSON and make one targeted repair attempt on schema failure."""
        working_messages = deepcopy(messages)
        last_error = None
        for attempt in range(2):
            response = completion(
                model=self.model,
                messages=working_messages,
                api_key=self.api_key,
                response_format={"type": "json_object"},
                num_retries=3,
            )
            raw = response.choices[0].message.content
            try:
                parsed = json.loads(raw)
                validated = schema.model_validate(parsed)
                return validated.model_dump()
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                if attempt == 0:
                    working_messages.extend([
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "The JSON failed schema validation. Correct only the structure and invalid values, "
                                f"preserving the therapeutic intent. Validation errors: {exc}"
                            ),
                        },
                    ])
        raise ValueError(f"LLM plan failed schema validation after repair: {last_error}")

    def _clinical_reasoning(self, user_input):
        """
        Backward-compatible single-shot clinical reasoning.
        This is still kept for API calls that do not use the DialogTherapistAgent.
        For the new UI pipeline, prefer get_intervention_plan_from_state().
        """
        reasoning_prompt = """
        You are a professional Psychologist and Restorative Environment Analyst/Strategist.
        Analyze the user's stress based on input.

        According to SRT (Stress Reduction Theory) and ART (Attention Restoration Theory):
        1. Identify the stress type (e.g., Cognitive Fatigue, High Anxiety, Seasonal Depression, Burnout, Loneliness, Mixed).
        2. Define target environmental features:
           - Lighting condition: warmth/intensity/brightness.
           - Complexity level.
           - Key psychological elements: Prospect, Refuge, Soft Fascination, Being Away, Extent, Compatibility.
        3. Give a clinical stress analysis based on the user input.
        4. Generate a search query that combines professional terms and visual targets.

        OUTPUT ONLY JSON:
        {{
          "stress_analysis": "...",
          "target_physics": {{"kelvin_range": "", "complexity": "", "...":"..."}},
          "search_query": "e.g., tranquil nature with golden hour warm lighting, low complexity, soft fascination elements"
        }}
        """

        user_context = [{"type": "text", "text": f"User input: {user_input}"}]

        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": reasoning_prompt},
                {"role": "user", "content": user_context}
            ],
            api_key=self.api_key,
            response_format={"type": "json_object"},
            num_retries=3
        )
        return json.loads(response.choices[0].message.content)

    def _clinical_reasoning_from_state(self, planning_context):
        """
        DialogTherapistAgent already estimates psychological state, so this method mainly
        converts it into an explicit planning strategy for RAG and generation.
        """
        psych_state = planning_context.get("psych_state", {}) or {}
        search_query = planning_context.get("search_query", "")
        dialog_summary = planning_context.get("dialog_summary", "")

        stress_type = psych_state.get("stress_type", "mixed")
        fatigue = psych_state.get("fatigue", 0.5)
        arousal = psych_state.get("arousal", 0.5)
        light_pref = psych_state.get("lighting_preference", "warm soft light")
        env_pref = psych_state.get("environment_preference", "restorative natural environment")
        sensory = psych_state.get("sensory_sensitivity", "moderate")

        if stress_type in ["high_arousal_anxiety", "anxiety"] or float(arousal or 0.5) >= 0.7:
            complexity = "low to moderate, predictable, non-chaotic"
            kelvin_range = "3000K-4000K"
            core = "increase safety/refuge, reduce harsh contrast and kinetic stimulation"
        elif stress_type in ["cognitive_fatigue", "burnout"] or float(fatigue or 0.5) >= 0.7:
            complexity = "moderate natural complexity with soft fascination"
            kelvin_range = "3200K-4500K"
            core = "increase being-away, spatial depth, soft fascination, and open prospect"
        elif stress_type in ["low_mood_sad", "loneliness"]:
            complexity = "low to moderate, gently uplifting"
            kelvin_range = "3500K-4800K"
            core = "increase safe brightness, warm sky exposure, gentle openness, and emotional warmth"
        else:
            complexity = "moderate restorative natural complexity"
            kelvin_range = "3200K-4500K"
            core = "balance prospect/refuge, soft fascination, and compatibility"

        return {
            "stress_analysis": (
                f"Dialog-guided psychological state suggests {stress_type}. "
                f"Arousal={arousal}, fatigue={fatigue}, sensory_sensitivity={sensory}. "
                f"Main intervention principle: {core}. Dialog summary: {dialog_summary}"
            ),
            "target_physics": {
                "kelvin_range": kelvin_range,
                "complexity": complexity,
                "preferred_environment": env_pref,
                "lighting_preference": light_pref,
                "avoid_elements": psych_state.get("avoid_elements", []),
                "target_srt_art_mechanisms": psych_state.get("target_srt_art_mechanisms", []),
                "greenery_ratio": "moderate to high unless user dislikes dense forest",
                "sky_ratio": "moderate to high when low mood or oppressive stress is present",
            },
            "search_query": search_query or f"{stress_type}, {env_pref}, {light_pref}, {core}",
            "psych_state": psych_state,
            "dialog_confidence": planning_context.get("dialog_confidence", 0.0),
        }

    def _path_from_filename(self, filename):
        if not filename:
            return None
        candidates = [
            os.path.join(self.picture_data_dir, filename),
            os.path.join("PictureData", filename),
            filename,
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return os.path.join(self.picture_data_dir, filename)

    def _reference_items_from_results(self, search_results):
        items = []
        docs = search_results.get("documents", [[]])[0]
        metas = search_results.get("metadatas", [[]])[0]
        distances = search_results.get("distances", [[]])[0]
        for i, meta in enumerate(metas):
            filename = meta.get("filename")
            items.append({
                "reference_index": i + 1,
                "reference_filename": filename,
                "reference_image_path": self._path_from_filename(filename),
                "reference_document": docs[i] if i < len(docs) else "",
                "reference_metadata": meta,
                "reference_distance": distances[i] if i < len(distances) else None,
            })
        return items

    @staticmethod
    def _kelvin_filter(target_kelvin, tolerance=800):
        try:
            target = float(target_kelvin)
        except (TypeError, ValueError):
            return None
        return {
            "$and": [
                {"estimated_kelvin": {"$gte": max(1500.0, target - tolerance)}},
                {"estimated_kelvin": {"$lte": min(12000.0, target + tolerance)}},
            ]
        }

    @staticmethod
    def _rerank_references(reference_items, target_kelvin=None):
        try:
            target = float(target_kelvin)
        except (TypeError, ValueError):
            target = None

        def score(ref):
            distance = ref.get("reference_distance")
            try:
                dense = float(distance)
            except (TypeError, ValueError):
                dense = 1.0
            kelvin_penalty = 0.0
            if target is not None:
                try:
                    kelvin_penalty = abs(float(ref["reference_metadata"].get("estimated_kelvin")) - target) / 4000.0
                except (TypeError, ValueError):
                    kelvin_penalty = 0.25
            return dense + kelvin_penalty

        return sorted(reference_items, key=score)

    def _attach_reference_images(self, plan, reference_items, exclude_filename=None):
        scenes = plan.get("scenes", []) if isinstance(plan, dict) else []
        if not scenes:
            return plan

        for scene in scenes:
            target_kelvin = scene.get("target_kelvin")
            try:
                scene_matches = self._query_references(
                    scene.get("image_prompt", "restorative natural environment"),
                    n_results=3,
                    metadata_filter=self._kelvin_filter(target_kelvin),
                )
            except Exception:
                scene_matches = []
            candidates = scene_matches or reference_items
            candidates = [r for r in candidates if r.get("reference_filename") != exclude_filename]
            candidates = self._rerank_references(candidates or reference_items, target_kelvin)
            refs = []
            for ref in candidates[:3]:
                refs.append({
                    "reference_index": ref["reference_index"],
                    "reference_filename": ref["reference_filename"],
                    "reference_image_path": ref["reference_image_path"],
                    "reference_metadata": ref["reference_metadata"],
                    "reference_distance": ref.get("reference_distance"),
                })

            scene["reference_images"] = refs
            if refs:
                scene["reference_filename"] = refs[0]["reference_filename"]
                scene["reference_image_path"] = refs[0]["reference_image_path"]
                scene["reference_index"] = refs[0]["reference_index"]
                scene["reference_metadata"] = refs[0]["reference_metadata"]

        return plan

    def _format_context(self, reference_items):
        context_items = []
        for ref in reference_items:
            meta = ref["reference_metadata"]
            item = (
                f"--- Reference Scene {ref['reference_index']} ---\n"
                f"Description: {ref['reference_document']}\n"
                f"Environment: {meta.get('environment', 'Unknown')}\n"
                f"Mood Tags: {meta.get('mood', 'N/A')}\n"
                f"Psychological: {meta.get('psychological', 'N/A')}\n"
                f"Physics: Kelvin={meta.get('estimated_kelvin', 6500):.0f}K, "
                f"Brightness={meta.get('brightness', 128):.1f}, "
                f"SkyRatio={meta.get('sky_ratio', 0):.2f}, "
                f"GreeneryRatio={meta.get('greenery_ratio', 0):.2f}, "
                f"Contrast={meta.get('contrast', 0):.1f}, "
                f"Complexity={meta.get('complexity', 0):.1f}, "
                f"FractalDimension={meta.get('fractal_dimension', 0):.2f}\n"
                f"Objects: {meta.get('objects', 'N/A')}\n"
                f"Filename: {ref['reference_filename']}"
            )
            context_items.append(item)
        return "\n".join(context_items)

    @staticmethod
    def _filter_from_clinical_insight(clinical_insight):
        value = str((clinical_insight.get("target_physics") or {}).get("kelvin_range", ""))
        numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", value)]
        if len(numbers) < 2:
            return None
        low, high = sorted(numbers[:2])
        return {"$and": [{"estimated_kelvin": {"$gte": low}}, {"estimated_kelvin": {"$lte": high}}]}

    def _query_references(self, query_text, n_results=8, metadata_filter=None):
        try:
            available = self.collection.count()
            n_results = min(max(1, int(n_results)), available) if available else 1
        except Exception:
            n_results = max(1, int(n_results))
        kwargs = {
            "query_texts": [query_text],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            kwargs["where"] = metadata_filter
        try:
            search_results = self.collection.query(**kwargs)
        except Exception:
            if not metadata_filter:
                raise
            kwargs.pop("where", None)
            search_results = self.collection.query(**kwargs)
        items = self._reference_items_from_results(search_results)
        if metadata_filter and not items:
            kwargs.pop("where", None)
            items = self._reference_items_from_results(self.collection.query(**kwargs))
        return items

    def _generate_plan(self, user_need, clinical_insight, reference_items):
        context_str = self._format_context(reference_items[:5])

        system_prompt = f"""
        You are a VR Stress Management Expert. You will receive a dialog-derived psychological state
        and matching examples from a validated 360° nature database.

        [CLINICAL STRATEGY]
        Stress analysis: {clinical_insight['stress_analysis']}
        Target features: {json.dumps(clinical_insight['target_physics'], ensure_ascii=False)}
        Dialog-derived psych state: {json.dumps(clinical_insight.get('psych_state', {}), ensure_ascii=False)}

        [DATABASE CONTEXT]
        {context_str}

        TASK: Create a 10-minute sequence of exactly 10 scenes and 2 related music tracks.

        SCIENTIFIC CONSTRAINTS:
        - Use the database examples as visual and environmental baselines.
        - Use SRT/ART targets from the dialog state instead of re-inferring the user from scratch.
        - Aim for 3000K-4500K for most stress relief scenes unless low mood requires slightly brighter warm daylight.
        - For high anxiety: reduce visual contrast, sharp edges, dense clutter, and kinetic cues; emphasize safety/refuge.
        - For cognitive fatigue/burnout: emphasize being-away, soft fascination, openness, moderate natural complexity, and spatial depth.
        - For low mood/SAD/loneliness: emphasize safe brightness, warm sky exposure, gentle openness, and emotionally warm natural cues.
        - Avoid overly bright colors, chaotic details, artificial urban density, vehicles, crowds, text, and aggressive structures.
        - Image prompt must be concise, ideally under 40 tokens.
        - Focus only on images. Do not include words like '360', 'panorama', or 'lighting' because ProductionAgent adds them automatically.
        - MUSIC GENERATION: exactly 2 instrumental tracks, each about 5 minutes, matching the scene condition.

        OUTPUT ONLY valid JSON:
        {{
            "scenes": [
                {{
                    "step": 1,
                    "duration": 60,
                    "image_prompt": "concise image prompt",
                    "target_kelvin": 3500,
                    "intensity": 1.0,
                    "therapeutic_goal": "..."
                }},
                {{
                    "step": 2,
                    "duration": 60,
                    "image_prompt": "concise image prompt",
                    "target_kelvin": 3800,
                    "intensity": 1.0,
                    "therapeutic_goal": "..."
                }},
                "... exactly 10 scene objects ..."
            ],
            "music_playlist": [
                {{"step": 1, "music_prompt": "...", "style": "...", "title": "..."}},
                {{"step": 2, "music_prompt": "...", "style": "...", "title": "..."}}
            ]
        }}
        """

        user_context = [{"type": "text", "text": f"User/Dialog planning context:\n{user_need}"}]

        plan = self._validated_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context},
            ],
            InterventionPlanSpec,
        )
        plan = self._attach_reference_images(plan, reference_items)
        return plan

    def get_intervention_plan(self, user_input):
        """Backward-compatible old entry point."""
        clinical_insight = self._clinical_reasoning(user_input)
        hybrid_query = f"User Need: {user_input}. Therapeutic Target: {clinical_insight.get('search_query', '')}"
        reference_items = self._query_references(
            hybrid_query,
            n_results=5,
            metadata_filter=self._filter_from_clinical_insight(clinical_insight),
        )
        plan = self._generate_plan(user_input, clinical_insight, reference_items)
        return plan, clinical_insight

    def get_intervention_plan_from_state(self, planning_context):
        """
        New dialog-guided entry point.
        planning_context should come from DialogTherapistAgent.build_planning_context().
        """
        clinical_insight = self._clinical_reasoning_from_state(planning_context)
        hybrid_query = (
            f"Dialog psychological state: {planning_context.get('dialog_summary', '')}. "
            f"Therapeutic target: {clinical_insight.get('search_query', '')}"
        )
        reference_items = self._query_references(
            hybrid_query,
            n_results=5,
            metadata_filter=self._filter_from_clinical_insight(clinical_insight),
        )
        user_need = json.dumps(planning_context, ensure_ascii=False, indent=2)
        plan = self._generate_plan(user_need, clinical_insight, reference_items)
        return plan, clinical_insight

    def refine_intervention_plan(
        self, original_scene, feedback, user_input, original_insight=None, psych_state=None,
        exclude_reference_filename=None,
    ):
        psych_state = psych_state or (original_insight or {}).get("psych_state", {}) or {}
        refinement_query = (
            f"User/Dialog Need: {user_input}. "
            f"Psych state: {json.dumps(psych_state, ensure_ascii=False)}. "
            f"Correction needed: {feedback.get('refinement_suggestion', '')}. "
            f"Avoid: {feedback.get('clinical_critique', '')}"
        )

        reference_items = self._query_references(refinement_query, n_results=3)
        context = [r.get("reference_document", "") for r in reference_items]

        system_prompt = f"""
        You are a Professional Strategy Refiner for restorative VR scenes.
        The previous image generation FAILED the therapist's audit: {feedback.get('clinical_critique', '')}

        [ORIGINAL CLINICAL GOAL]
        {json.dumps((original_insight or {}).get('target_physics', {}), ensure_ascii=False)}

        [DIALOG-DERIVED PSYCH STATE]
        {json.dumps(psych_state, ensure_ascii=False)}

        Based on the refinement suggestion and user's dialog-derived psychological state,
        rewrite the image prompt while preserving the therapeutic goal.

        STRICT RULES:
        1. Do not append feedback mechanically; rewrite into one cohesive sentence.
        2. Image prompt must be concise and under 40 tokens if possible.
        3. Keep important visual corrections at the beginning.
        4. Do not include words like '360', 'panorama', or 'lighting'.
        5. Keep the same step id and duration if possible.

        OUTPUT ONLY valid JSON:
        {{
            "scenes": [
                {{
                    "step": {original_scene.get('step', 1) if isinstance(original_scene, dict) else 1},
                    "duration": {original_scene.get('duration', 60) if isinstance(original_scene, dict) else 60},
                    "image_prompt": "rewritten concise image prompt",
                    "target_kelvin": 3500,
                    "intensity": 1.0,
                    "therapeutic_goal": "..."
                }}
            ]
        }}
        """

        user_context = [{
            "type": "text",
            "text": f"Previous scene: {original_scene}. New database context: {context}. Original/Dialog user input: {user_input}"
        }]
        plan = self._validated_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context},
            ],
            RefinementPlanSpec,
        )
        expected_step = original_scene.get("step", 1) if isinstance(original_scene, dict) else 1
        if plan["scenes"][0]["step"] != expected_step:
            plan["scenes"][0]["step"] = expected_step
        plan = self._attach_reference_images(plan, reference_items, exclude_filename=exclude_reference_filename)
        return plan
