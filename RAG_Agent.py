import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import json

class RAGAgent:
    def __init__(self, api_key, db_path="./PictureBase"):
        self.client = OpenAI(api_key=api_key)
        self.emb_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
        self.db_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.db_client.get_collection(
            name="nature_environments",
            embedding_function=self.emb_fn
        )

    def get_intervention_plan(self, user_input):
        """
        Agent core logic:
        -> Retrieve similar scenarios and their physical metadata
        -> Make a comprehensive decision
        -> Output a JSON plan
        """

        # RAG Search
        search_results = self.collection.query(
            query_texts=[user_input],
            n_results=10,
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
                f"Physics: Kelvin={meta.get('estimated_kelvin', 6500):.0f}K, "
                f"Brightness={meta.get('brightness', 128):.1f}, "
                f"Colorfulness={meta.get('colorfulness', 0):.1f}\n"
                f"Sharpness={meta.get('sharpness', 0):.1f}\n"
                f"Contrast={meta.get('contrast', 0):.1f}\n"
                f"Objects: {meta.get('objects', 'N/A')}"
                f"  Filename: {meta['filename']}"
            )
            context_items.append(item)

        context_str = "\n".join(context_items)

        # Define the system prompt
        system_prompt = f"""
        You are a VR Stress Management Expert. You will receive a student's stress description 
        and matching examples from our validated 360° nature database - Database Context:\n{context_str}.

        TASK: Create a 10-minute (600s) sequence of 15 scenes and 2 relative music.

        SCIENTIFIC CONSTRAINTS:
        - Use the information from the database examples as a baseline for new generation.
        - Specifically, aim for 3000K-4500K for virtual sunlight to maximize anxiolytic effects(Kelvin number should better be mentioned in each image prompt).
        - If user mentions 'dark', 'winter', or 'low energy' that indicate they need more brightness, prioritize 'Warm' lighting scenes.
        - Compare the 'Original Kelvin' of the reference image. If it's too high (cool), 
          instruct the VR system to override it with a target_kelvin between 3200K-4000K.
        - Avoid using overly bright colors which can cause colorful chaos.
        - Your outputs should refer to 'Database Context' in some extents.
        - Image prompt must be highly concise (STRICTLY MAX 25 WORDS).
        - Focus only on images. Do not include words like '360', 'panorama', or 'lighting' here as they are added automatically.        
        - MUSIC GENERATION: You must suggest exactly 2 music tracks for the 300s session, each track should last about 150s.
        - Provide a 'music_prompt' for Suno API. It should be instrumental, focused on relaxation (e.g., "Ambient piano with soft wind sounds, 432Hz, meditative, slow tempo").
        - The music provided should be related to image conditions.(e.g., 'Warm Ambient' for 'Warm' lighting)
        - Target: Stress relief for users.

        OUTPUT FORMAT: Return ONLY a JSON object with:
        {{
            "scenes": [
                {{
                    "step": 1,
                    "duration": 60,
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
        user_context = f"Student Input: {user_input}\n\n"

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context}
            ],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)




