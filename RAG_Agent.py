import chromadb
from chromadb.utils import embedding_functions
from litellm import completion
import json

class RAGAgent:
    def __init__(self, api_key,  chroma_api, model_name="gpt-5.4", db_path="./PictureBase"):
        self.api_key = api_key
        self.model = model_name
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
        You are a professional Psychologist and Clinical Environment Analyst/Strategist. 
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
            num_retries= 3
        )
        return json.loads(response.choices[0].message.content)

    def get_intervention_plan(self, user_input):
        """
        Agent core logic:
        -> Retrieve similar scenarios and their physical metadata
        -> Make a comprehensive decision
        -> Output a JSON plan
        """

        clinical_insight = self._clinical_reasoning(user_input)
        hybrid_query = f"User Need: {user_input}. Therapeutic Target: {clinical_insight.get('search_query', '')}"

        # RAG Search
        search_results = self.collection.query(
            query_texts=[hybrid_query],
            n_results=8,
            include=["documents", "metadatas"]
        )

        # Binding textual descriptions to quantization parameters
        context_items = []
        for i in range(len(search_results['documents'][0])):
            doc = search_results['documents'][0][i]
            meta = search_results['metadatas'][0][i]
            item = (
                f"--- Reference Scene {i + 1} ---\n"
                f"Description: {search_results['documents'][0][i]}\n"
                f"Environment: {meta.get('environment', 'Unknown')}\n"
                f"Mood Tags: {meta.get('mood', 'N/A')}\n"
                f"Psychological: {meta.get('psychological', 'N/A')}\n"
                f"Physics: Kelvin={meta.get('estimated_kelvin', 6500):.0f}K, "
                f"Brightness={meta.get('brightness', 128):.1f}, "
                f"SkyRatio={meta.get('sky_ratio', 0):.2f}\n"
                f"GreeneryRatio={meta.get('greenery_ratio', 0):.2f}\n"
                f"Contrast={meta.get('contrast', 0):.1f}\n"
                f"Complexity={meta.get('complexity', 0):.1f}\n"
                f"FractalDimension={meta.get('fractal_dimension', 0):.2f}\n"
                f"Objects: {meta.get('objects', 'N/A')}"
                f"  Filename: {meta['filename']}"
            )
            context_items.append(item)

        context_str = "\n".join(context_items)

        # Define the system prompt
        system_prompt = f"""
        You are a VR Stress Management Expert. You will receive a student's stress description
        and matching examples from our validated 360° nature database:
        [CLINICAL STRATEGY]: 
        Based on analysis, the user is experiencing {clinical_insight['stress_analysis']}. 
        Target features: {json.dumps(clinical_insight['target_physics'])}.

        [DATABASE CONTEXT]: {context_str}

        TASK: Create a 10-minute (600s) sequence of 5 scenes and 2 relative music.

        SCIENTIFIC CONSTRAINTS:
        - Use the information from the database examples as a baseline for new generation.
        - Specifically, aim for 3000K-4500K for virtual sunlight to maximize anxiolytic effects(Kelvin number or lighting condition should better be mentioned in each image prompt).
        - If user mentions 'dark', 'winter', or 'low energy' that indicate they need more brightness, prioritize 'Warm' lighting scenes.
        - Compare the 'Original Kelvin' of the reference image. If it's too high (cool), 
          instruct the VR system to override it with a target_kelvin between 3200K-4000K.
        - Avoid using overly bright colors which can cause colorful chaos.
        - Your outputs should refer to 'Database Context' in some extents.
        - Image prompt must be highly concise (NOT exceeding 40 tokens).
        - Focus only on images. Do not include words like '360', 'panorama', or 'lighting' here as they are added automatically.        
        - MUSIC GENERATION: You must suggest exactly 2 music tracks for the 300s session, each track should last about 150s.
        - Provide a 'music_prompt' for Suno API. It should be instrumental, focused on relaxation (e.g., "Ambient piano with soft wind sounds, 432Hz, meditative, slow tempo").
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

        # Execution reasoning
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
            api_key= self.api_key,
            response_format={"type": "json_object"},
            num_retries=3
        )

        return json.loads(response.choices[0].message.content), clinical_insight

    def refine_intervention_plan(self, original_scene, feedback, user_input, original_insight=None):
        """
        Skill: Strategic Correction.
        Re-queries the database based on the therapist's failure feedback.
        """
        refinement_query = (
            f"User Original Need: {user_input}. "
            f"Correction needed: {feedback.get('refinement_suggestion', '')}. "
            f"Focus on environmental features that avoid: {feedback.get('clinical_critique', '')}"
        )

        # Re-search the vector DB for a more suitable environment
        results = self.collection.query(query_texts=[refinement_query], n_results=3)
        context = results['documents'][0]

        system_prompt = f"""
        You are a Clinical Strategy Refiner. The previous image generation FAILED the therapist's audit:{feedback['clinical_critique']}
        [ORIGINAL CLINICAL GOAL]: {json.dumps(original_insight.get('target_physics', {})) if original_insight else "None"}
        Based on the refinement suggestion:{feedback['refinement_suggestion']} and user's original input, 
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
            api_key= self.api_key,
            response_format={"type": "json_object"},
            num_retries=3
        )
        # Returns a single refined scene object
        return json.loads(response.choices[0].message.content)




