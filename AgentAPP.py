import os
import time
import json
import uvicorn
import shutil
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

    def __init__(self, model_name="gpt-5.4", shared_production=None, shared_filter=None):
        self.model_name = model_name

        if "gpt" in model_name.lower():
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif "claude" in model_name.lower():
            self.api_key = os.getenv("CLAUDE_API_KEY")
        elif "gemini" in model_name.lower():
            self.api_key = os.getenv("GEMINI_API_KEY")
        elif "deepseek" in model_name.lower():
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
        else:
            self.api_key = os.getenv("API_KEY")
        self.hf_token = os.getenv("HF_TOKEN")
        self.openai_api = os.getenv("CHROMA_OPENAI_API_KEY")

        self.rag = RAGAgent(self.api_key, self.openai_api, model_name=model_name)
        self.therapist = TherapistAgent(self.api_key, self.openai_api, model_name=model_name)
        if shared_production:
            self.production = shared_production
        else:
            self.production = ProductionAgent(
                suno_api_key=os.getenv("SUNO_API_KEY"),
                suno_base_url=os.getenv("SUNO_API_BASE")
            )

        if shared_filter:
            self.filter = shared_filter
        else:
            self.filter = FilterAgent(hf_token=self.hf_token)
        self.MAX_AUDIT_RETRIES = 3

    async def execute(self, user_input: str, case_id: int = 0):
        """
        RAG - Production - Filter
        """
        safe_model_name = self.model_name.replace("/", "_").replace("\\", "_")
        case_root = f"static/results/{safe_model_name}/case_{case_id}"
        path_base = os.path.join(case_root, "base")
        path_audit = os.path.join(case_root, "audit")
        path_case_final = os.path.join(case_root, "final_images")
        path_global_final = "static/final_images_all"
        for p in [path_base, path_audit, path_case_final, path_global_final]:
            os.makedirs(p, exist_ok=True)

        session_logs = {
            "model_name": self.model_name,
            "start_time": time.time(),
            "user_input": user_input,
            "clinical_insight": None,
            "audit_retries": 0,
            "iteration_history": [],  #  Prompt, Metrics, Audit
        }

        print("🔗 [Chain] Step 1: Formulating strategy via RAG...")
        plan, clinical_insight = self.rag.get_intervention_plan(user_input)
        session_logs["clinical_insight"] = clinical_insight

        print("🔗 [Chain] Step 2: Generating healing music and scenes")
        music_task = [asyncio.to_thread(self.production.generate_music,
                                        t['music_prompt'], t['style'], t['title'])
                      for t in plan.get('music_playlist', [])]

        scene_tasks = [self.production.generate_image(
            s['image_prompt'],
            s['step'],
            folder=path_base,
            filename=f"step_{s['step']}_init.jpg"
        ) for s in plan.get('scenes', [])]

        for s in plan.get('scenes', []):
            session_logs["iteration_history"].append({
                "round": 0,
                "step": s['step'],
                "prompt": s['image_prompt'],
                "audit": None,
                "metrics": None
            })
        all_results = await asyncio.gather(*music_task, *scene_tasks)

        music_urls = all_results[:len(music_task)]
        scene_results = all_results[len(music_task):]

        music_resources = []
        for i, url in enumerate(music_urls):
            track = plan['music_playlist'][i]
            safe_url = url if url else "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
            music_resources.append({"step": track['step'], "title": track['title'], "audio_url": safe_url})

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
                    user_input,
                    clinical_insight
                )

                results_map[step_id]["audit_report"] = audit_res
                results_map[step_id]["metrics"] = metrics

                for entry in session_logs["iteration_history"]:
                    if entry["round"] == retry_count and entry["step"] == step_id:
                        entry["audit"] = audit_res
                        entry["metrics"] = metrics

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
            session_logs["audit_retries"] = retry_count

            for step_id in steps_to_fix:
                old_data = results_map[step_id]
                refined_plan = self.rag.refine_intervention_plan(
                    old_data["scene_data"],
                    old_data["audit_report"],
                    user_input,
                    original_insight=clinical_insight
                )
                session_logs["iteration_history"].append({
                    "round": retry_count,
                    "step": step_id,
                    "prompt": refined_plan['scenes'][0]['image_prompt'],
                    "audit": None,
                    "metrics": None
                })

                _, new_img_path = await self.production.generate_image(
                    refined_plan['scenes'][0]['image_prompt'],
                    step_id,
                    folder=path_audit,
                    filename=f"step_{step_id}_retry_{retry_count}.jpg"
                )

                if new_img_path:
                    results_map[step_id]["image_path"] = new_img_path
                    results_map[step_id]["scene_data"] = refined_plan['scenes'][0]

        final_scenes = []
        for step_id in sorted(results_map.keys()):
            data = results_map[step_id]
            identifier = f"{safe_model_name}_c{case_id}_s{step_id}"
            case_final_path = self.production.upscale_image(
                data["image_path"],
                step_id,
                folder=path_case_final
            )
            global_final_filename = f"{identifier}_final.png"
            global_final_path = os.path.join(path_global_final, global_final_filename)
            try:
                shutil.copy2(case_final_path, global_final_path)
            except Exception as e:
                print(f"⚠️ Global Copy Error: {e}")

            final_scenes.append({
                "step": step_id,
                "image_path": case_final_path,
                "image_url": f"http://localhost:8000/{case_final_path.replace(os.sep, '/')}",
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
            "status": "Success",
            "model_used": self.model_name
        }
        return final_json, session_logs

app = FastAPI()
latest_session_result = None

app.mount("/static", StaticFiles(directory="static"), name="static")

shared_prod = None
shared_filt = None
def get_shared_agents():
    global shared_prod, shared_filt
    if shared_prod is None:
        shared_prod = ProductionAgent(
            suno_api_key=os.getenv("SUNO_API_KEY"),
            suno_base_url=os.getenv("SUNO_API_BASE")
        )
    if shared_filt is None:
        shared_filt = FilterAgent(hf_token=os.getenv("HF_TOKEN"))
    return shared_prod, shared_filt

class UserRequest(BaseModel):
    description: str
    model_name: str = "gpt-5.4"

@app.get("/get-latest-session")
async def get_latest_session():
    return latest_session_result


@app.post("/generate-session")
async def create_session(request: UserRequest):
    global latest_session_result
    try:
        prod, filt = get_shared_agents()
        chain = TherapyChain(model_name=request.model_name, shared_production=prod, shared_filter=filt)
        final_data, logs = await chain.execute(request.description)
        latest_session_result = final_data
        return {"data": final_data, "logs": logs}
    except Exception as e:
        return {"error": str(e), "status": "failed"}

async def intervention_gui_logic(user_input):
    global latest_session_result
    if not user_input.strip():
        return [], None, gr.update(choices=[]), "Input is empty."

    try:
        prod, filt = get_shared_agents()
        chain = TherapyChain(model_name="gpt-5.4", shared_production=prod, shared_filter=filt)
        result, logs = await chain.execute(user_input)

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
