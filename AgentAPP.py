import os
import json
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import gradio as gr
import asyncio
from RAG_Agent import RAGAgent
from Production_Agent import ProductionAgent
from Filter_Agent import FilterAgent
from Therapsit_Agent import TherapistAgent

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
load_dotenv()
API_KEY = os.getenv("API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


class TherapyChain:

    def __init__(self, rag_agent, production_agent, filter_agent, therapist_agent):
        self.rag = rag_agent
        self.production = production_agent
        self.filter = filter_agent
        self.therapist = therapist_agent
        self.MAX_AUDIT_RETRIES = 2

    async def execute(self, user_input: str):
        """
        RAG - Production - Filter
        """
        print("🔗 [Chain] Step 1: Formulating strategy via RAG...")
        plan, clinical_insight = self.rag.get_intervention_plan(user_input)

        print("🔗 [Chain] Step 2: Generating healing music and scenes")
        music_task = [asyncio.to_thread(self.production.generate_music,
                                        t['music_prompt'], t['style'], t['title'])
                      for t in plan.get('music_playlist', [])]

        scene_tasks = [self.production.generate_image(
            s['image_prompt'], s['step'], folder="static/base")
            for s in plan.get('scenes', [])]

        all_results = await asyncio.gather(*music_task, *scene_tasks)

        music_urls = all_results[:len(music_task)]
        scene_results = all_results[len(music_task):]

        music_resources = []
        for i, url in enumerate(music_urls):
            track = plan['music_playlist'][i]
            music_resources.append({"step": track['step'], "title": track['title'], "audio_url": url})

        results_map = {}
        for i, res in enumerate(scene_results):
            _, local_path = res
            scene = plan['scenes'][i]
            results_map[scene['step']] = {
                "scene_data": scene, "image_path": local_path,
                "is_passed": False, "audit_report": {}, "metrics": {}
            }

        print("🔗 [Chain] Step 3: Auditing candidate scenes...")
        retry_count = 0
        while retry_count <= self.MAX_AUDIT_RETRIES:
            all_passed = True
            print(f"\n🩺 [Chain] Audition {retry_count} ")

            steps_to_fix = []

            for step_id, data in results_map.items():
                if data["is_passed"]:
                    continue

                metrics = self.filter.get_physical_report(data["image_path"])
                audit_res = self.therapist.audit_scene(
                    data["image_path"],
                    data["scene_data"],
                    metrics,
                    user_input
                )

                results_map[step_id]["audit_report"] = audit_res
                results_map[step_id]["metrics"] = metrics

                if audit_res.get("decision") == "PASS":
                    results_map[step_id]["is_passed"] = True
                    print(f"✅ Step {step_id}: Passed")
                else:
                    all_passed = False
                    steps_to_fix.append(step_id)
                    print(f"❌ Step {step_id}: Failed - {audit_res.get('clinical_critique')}")

            if all_passed or retry_count >= self.MAX_AUDIT_RETRIES:
                break

            retry_count += 1

            for step_id in steps_to_fix:
                old_data = results_map[step_id]
                refined_plan = self.rag.refine_intervention_plan(
                    old_data["scene_data"],
                    old_data["audit_report"],
                    user_input,
                    original_insight=clinical_insight
                )
                _, new_img_path = await self.production.generate_image(
                    refined_plan['scenes'][0]['image_prompt'],
                    f"{step_id}_retry_{retry_count}",
                    folder="static/audit"
                )

                results_map[step_id]["image_path"] = new_img_path
                results_map[step_id]["scene_data"] = refined_plan['scenes'][0]

        final_scenes = []
        for step_id in sorted(results_map.keys()):
            data = results_map[step_id]
            final_img_upscaled = self.production.upscale_image(data["image_path"], step_id)
            final_scenes.append({
                "step": step_id,
                "image_path": final_img_upscaled,
                "image_url": f"http://localhost:8000/{final_img_upscaled}",
                "duration": data["scene_data"].get("duration", 60),
                "audit_critique": data["audit_report"].get("clinical_critique", "Final version"),
                "unity_config": {
                    "kelvin": data["metrics"].get("estimated_kelvin", 4000),
                    "intensity": data["scene_data"].get("intensity", 1.0)
                }
            })

        final_json = {
            "user_input": user_input,
            "intervention_plan": final_scenes,
            "music_playlist": music_resources,
            "status": "Success"
        }
        return final_json

app = FastAPI()
latest_session_result = None

rag_agent = RAGAgent(API_KEY)
production_agent = ProductionAgent(os.getenv("SUNO_API_KEY"), os.getenv("SUNO_API_BASE"))
filter_agent = FilterAgent(API_KEY, HF_TOKEN)
therapist_agent = TherapistAgent(API_KEY)

therapy_chain = TherapyChain(rag_agent, production_agent, filter_agent, therapist_agent)

STATIC_DIRS = ["static/base", "static/audit", "static/final_images"]
for folder in STATIC_DIRS:
    os.makedirs(folder, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


class UserRequest(BaseModel):
    description: str

@app.get("/get-latest-session")
async def get_latest_session():
    return latest_session_result


@app.post("/generate-session")
async def create_session(request: UserRequest):
    """API interface for Unity"""
    global latest_session_result
    try:
        final_data = await therapy_chain.execute(request.description)
        latest_session_result = final_data
        return final_data
    except Exception as e:
        return {"error": str(e), "status": "failed"}

async def intervention_gui_logic(user_input):
    global latest_session_result
    if not user_input.strip():
        return [], None, gr.update(choices=[]), "Input is empty."

    try:
        result = await therapy_chain.execute(user_input)
        latest_session_result = result

        gallery_data = [(s['image_path'], f"Step {s['step']}: {s['audit_critique'][:30]}...") for s in
                        result['intervention_plan']]

        audio_urls = [m['audio_url'] for m in result['music_playlist']]
        initial_audio = audio_urls[0] if audio_urls else None

        return (
            gallery_data,
            initial_audio,
            gr.update(choices=audio_urls, value=initial_audio),
            json.dumps(result, indent=4, ensure_ascii=False)
        )
    except Exception as e:
        return [], None, gr.update(choices=[]), f"Error: {str(e)}"


custom_css = """
#main-container { background-color: #0b0f19; }
.gradio-container { font-family: 'Segoe UI', sans-serif; }
#title-text { text-align: center; color: #4f46e5; margin-bottom: 20px; }
#gallery-container { border: 2px solid #4f46e5; border-radius: 12px; padding: 10px; background: #111827; height: auto; }
.audio-group { background: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="#0b0f19",
    block_background_fill="#111827",
    block_label_text_color="#94a3b8"
)

with gr.Blocks(theme=theme, css=custom_css, title="LLM-Driven VR Art Therapy") as demo:
    gr.Markdown("# 🎨 VR Personalized Art Intervention Agent", elem_id="title-text")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🧠 Emotion Input")
            input_text = gr.Textbox(
                label="Describe your stress or feelings",
                placeholder="For example: Swedish winters are so dark, I feel very lonely...",
                lines=5
            )
            submit_btn = gr.Button("✨ Generate a healing plan (image + audio)", variant="primary")
            gr.Markdown("---")
            with gr.Group(elem_classes="audio-group"):
                gr.Markdown("### 🎵 Healing Audio (Suno AI)")
                audio_player = gr.Audio(label="Playing", interactive=False, type="filepath")
                music_selector = gr.Dropdown(
                    label="Switch song",
                    choices=[],
                    interactive=True
                )

        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ Generate a results gallery (click to preview/switch)")
            image_gallery = gr.Gallery(
                label="Generated scene sequences",
                show_label=False,
                elem_id="gallery-container",
                columns=[2],
                rows=[2],
                object_fit="contain",
                height="auto"
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 System Instruction (JSON)")
            json_output = gr.Code(label="Unity Data Stream", language="json")

    # Bind event
    submit_btn.click(
        fn=intervention_gui_logic,
        inputs=input_text,
        outputs=[image_gallery, audio_player, music_selector, json_output]
    )

    # Update the player address when the selection box changes
    music_selector.change(
        fn=lambda x: x,
        inputs=music_selector,
        outputs=audio_player
    )

app = gr.mount_gradio_app(app, demo, path="/gui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=65)
