import os
import json
import chromadb
from chromadb.utils import embedding_functions
from litellm import completion


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

    def _clinical_reasoning(self, user_input):
        """
        Transform user input to clinical index
        """
        reasoning_prompt = f"""
        You are a professional Psychologist and Restorative Environment Analyst/Strategist. 
        Analyze the user's stress based on input.

        According to SRT (Stress Reduction Theory) and ART (Attention Restoration Theory):
        1. Identify the stress type (e.g., Cognitive Fatigue, High Anxiety, Seasonal Depression .... - Define this by yourself).
        2. Define target environmental features:
           - Lighting condition: Such as warmth or intensity.
           - Complexity Level
           - Key Psychological Elements: (e.g., Prospect, Refuge, Soft Fascination ..... - Define this by yourself.)
        3. Give a clinical stress analysis based on your psychological knowledge and analysis to user input.
        4. Generate a 'Search Query' that combines these professional terms.

        OUTPUT ONLY JSON:
        {{
          "stress_analysis": "...",
          "target_physics": {{"kelvin_range": "", "complexity": "", "...":"..."}},
          "search_query": "e.g., tranquil nature with golden hour warm lighting, low complexity, soft fascination elements"
        }}
        The target_physics contains any elements that you think is helpful for RAG agent to search.
        """

        user_context = [
            {
                "type": "text",
                "text": f"User input: {user_input}"
            }]

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
        for i, meta in enumerate(metas):
            filename = meta.get("filename")
            items.append({
                "reference_index": i + 1,
                "reference_filename": filename,
                "reference_image_path": self._path_from_filename(filename),
                "reference_document": docs[i] if i < len(docs) else "",
                "reference_metadata": meta,
            })
        return items

    def _attach_reference_images(self, plan, reference_items):
        scenes = plan.get("scenes", []) if isinstance(plan, dict) else []
        if not scenes or not reference_items:
            return plan

        for idx, scene in enumerate(scenes):
            start = idx % len(reference_items)

            refs = []
            for j in range(3):
                ref = reference_items[(start + j) % len(reference_items)]
                refs.append({
                    "reference_index": ref["reference_index"],
                    "reference_filename": ref["reference_filename"],
                    "reference_image_path": ref["reference_image_path"],
                    "reference_metadata": ref["reference_metadata"],
                })

            scene["reference_images"] = refs

            scene["reference_filename"] = refs[0]["reference_filename"]
            scene["reference_image_path"] = refs[0]["reference_image_path"]
            scene["reference_index"] = refs[0]["reference_index"]
            scene["reference_metadata"] = refs[0]["reference_metadata"]

        return plan

    def get_intervention_plan(self, user_input):
        clinical_insight = self._clinical_reasoning(user_input)
        hybrid_query = f"User Need: {user_input}. Therapeutic Target: {clinical_insight.get('search_query', '')}"

        search_results = self.collection.query(
            query_texts=[hybrid_query],
            n_results=8,
            include=["documents", "metadatas"]
        )
        reference_items = self._reference_items_from_results(search_results)

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
        context_str = "\n".join(context_items)

        system_prompt = f"""
        You are a VR Stress Management Expert. You will receive a student's stress description
        and matching examples from our validated 360° nature database:

        [CLINICAL STRATEGY]
        Stress analysis: {clinical_insight['stress_analysis']}
        Target features: {json.dumps(clinical_insight['target_physics'])}

        [DATABASE CONTEXT]
        {context_str}

        TASK: Create a 10-minute sequence of 2 scenes and 2 relative music tracks.

        SCIENTIFIC CONSTRAINTS:
        - Use the information from the database examples as a baseline for new generation.
        - Specifically, aim for 3000K-4500K for virtual sunlight to maximize anxiolytic effects
          (Kelvin number or lighting condition should better be mentioned in each image prompt).
        - If user mentions 'dark', 'winter', or 'low energy' that indicate they need more brightness, prioritize 'Warm' lighting scenes.
        - Compare the 'Original Kelvin' of the reference image. If it's too high (cool), 
          instruct the VR system to override it with a target_kelvin between 3200K-4000K.
        - Avoid using overly bright colors which can cause colorful chaos.
        - Your outputs should refer to 'Database Context' in some extents.
        - Image prompt must be highly concise (NOT exceeding 40 tokens).
        - Focus only on images. Do not include words like '360', 'panorama', or 'lighting' here as they are added automatically.        
        - MUSIC GENERATION: You must suggest exactly 2 music tracks for the 300s session, each track should last about 150s.
        - Provide a 'music_prompt' for Suno API. It should be instrumental, focused on relaxation 
          (e.g., "Ambient piano with soft wind sounds, 432Hz, meditative, slow tempo").
        - The music provided should be related to image conditions.(e.g., 'Warm Ambient' for 'Warm' lighting)
        - Target: Stress relief for users.

        OUTPUT FORMAT: Return ONLY a valid **JSON** object with:
        {{
            "scenes": [
                {{
                    "step": 1,
                    "duration": (e.g., 60),
                    "image_prompt": "detailed prompt for image generation model",
                    "target_kelvin": (e.g., 3500),
                    "intensity": (e.g., 1.2)), 
                }}
            ]
            "music_playlist": [
            {{ "step": 1, "music_prompt": "...", "style": "...", "title": "..." }},
            {{ "step": 2, "music_prompt": "...", "style": "...", "title": "..." }}
            ]
        }}
        """

        user_context = [
            {
                "type": "text",
                "text": f"User Input: {user_input}\n\n",
            }
        ]

        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context}
            ],
            api_key=self.api_key,
            response_format={"type": "json_object"},
            num_retries=3
        )
        plan = json.loads(response.choices[0].message.content)
        plan = self._attach_reference_images(plan, reference_items)
        return plan, clinical_insight

    def refine_intervention_plan(self, original_scene, feedback, user_input, original_insight=None):
        refinement_query = (
            f"User Original Need: {user_input}. "
            f"Correction needed: {feedback.get('refinement_suggestion', '')}. "
            f"Focus on environmental features that avoid: {feedback.get('clinical_critique', '')}"
        )

        results = self.collection.query(
            query_texts=[refinement_query],
            n_results=3,
            include=["documents", "metadatas"]
        )
        reference_items = self._reference_items_from_results(results)
        context = results.get("documents", [[]])[0]

        system_prompt = f"""
        You are a Professional Strategy Refiner.
        The previous image generation FAILED the therapist's audit: {feedback.get('clinical_critique', '')}

        [ORIGINAL CLINICAL GOAL]
        {json.dumps(original_insight.get('target_physics', {})) if original_insight else "None"}

        Based on the refinement suggestion: {feedback.get('refinement_suggestion', '')} and user's original input, 
        your task is to REWRITE the image prompt to fix the issues while staying under the token limit.

        STRICT RULES:
        1. COMPRESSION: Do not simply add feedback to the old prompt. REWRITE it into a single, cohesive sentence.
        2. TOKEN LIMIT: Image prompt must be concise. The total prompt MUST NOT exceed 40 tokens to ensure compatibility with CLIP.
        3. REMOVE REDUNDANCY: Delete flowery adjectives. (Instead of "a sense of calm and comforting stillness," use "serene, tranquil.")
        4. STRUCTURE: Keep the most important visual changes at the BEGINNING. Focus only on images. Do not include words like '360', 
        'panorama', or 'lighting' here as they are added automatically. 

        The output format should remain the same **JSON** format:
        {{
            "scenes": [
                {{
                    "step": 1,
                    "duration": (e.g., 60),
                    "image_prompt": "detailed prompt for image generation model",
                    "target_kelvin": (e.g., 3500),
                    "intensity": (e.g., 1.2)), 
                }}
            ]
        }}     
        """

        user_context = [
            {
                "type": "text",
                "text": f"Previous: {original_scene}. New Context from database: {context}. Original user input: {user_input}"
            }
        ]
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context}
            ],
            api_key=self.api_key,
            response_format={"type": "json_object"},
            num_retries=3
        )
        plan = json.loads(response.choices[0].message.content)
        plan = self._attach_reference_images(plan, reference_items)
        return plan
