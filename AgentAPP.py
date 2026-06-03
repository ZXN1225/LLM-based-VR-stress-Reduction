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
from MetricsToolBox import ToolBox
from Auditing_Agent import AuditingAgent

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
        self.therapist = AuditingAgent(self.api_key, self.openai_api, model_name=model_name)
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
            self.filter = ToolBox(hf_token=self.hf_token)
        self.MAX_AUDIT_RETRIES = 3

    async def execute(self, user_input: str, case_id: int = 0):
        """
        RAG -> Production -> Auditing.

        This version uses a flat result directory structure:
            static/results/base
            static/results/audit
            static/results/final_images
            static/results/supir_input
            static/results/supir_output

        Important:
        - SUPIR is NOT executed inside AgentAPP.
        - After auditing, final images are copied to final_images/ and supir_input/.
        - Close AgentAPP, then run SUPIR separately on supir_input/ to avoid GPU conflicts.
        """
        result_root = "static/results"
        path_base = os.path.join(result_root, "base")
        path_audit = os.path.join(result_root, "audit")
        path_case_final = os.path.join(result_root, "final_images")
        path_supir_input = os.path.join(result_root, "supir_input")
        path_supir_output = os.path.join(result_root, "supir_output")

        for p in [path_base, path_audit, path_case_final, path_supir_input, path_supir_output]:
            if os.path.exists(p):
                shutil.rmtree(p)
            os.makedirs(p, exist_ok=True)

        session_logs = {
            "model_name": self.model_name,
            "start_time": time.time(),
            "user_input": user_input,
            "clinical_insight": None,
            "audit_retries": 0,
            "iteration_history": [],
        }

        print("🔗 [Chain] Step 1: Formulating strategy via RAG...")
        plan, clinical_insight = self.rag.get_intervention_plan(user_input)
        session_logs["clinical_insight"] = clinical_insight

        print("🔗 [Chain] Step 2: Generating healing music and scenes")
        music_task = [
            asyncio.to_thread(
                self.production.generate_music,
                t['music_prompt'],
                t['style'],
                t['title']
            )
            for t in plan.get('music_playlist', [])
        ]

        scene_tasks = [
            self.production.generate_image(
                s['image_prompt'],
                s['step'],
                folder=path_base,
                filename=f"step_{s['step']}_init.jpg",
                ref_image_path=s.get("reference_image_path"),
                scene_data=5
            )
            for s in plan.get('scenes', [])
        ]

        all_results = await asyncio.gather(*music_task, *scene_tasks)

        music_urls = all_results[:len(music_task)]
        scene_results = all_results[len(music_task):]

        music_resources = []
        for i, url in enumerate(music_urls):
            track = plan['music_playlist'][i]
            safe_url = url if url else "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
            music_resources.append({
                "step": track['step'],
                "title": track['title'],
                "audio_url": safe_url
            })

        results_map = {}
        for i, res in enumerate(scene_results):
            scene = plan['scenes'][i]
            step_id = scene['step']

            if not res or not isinstance(res, tuple) or len(res) < 2:
                print(f"⚠️ Scene {step_id} generation returned invalid result; skipping.")
                continue

            _, local_path = res
            if local_path is None or not os.path.exists(local_path):
                print(f"⚠️ Scene {step_id} generation failed; skipping audit for this scene.")
                continue

            results_map[step_id] = {
                "scene_data": scene,
                "reference_image_path": scene.get("reference_image_path"),
                "reference_filename": scene.get("reference_filename"),
                "image_path": local_path,
                "is_passed": False,
                "audit_report": {},
                "metrics": {},
                "last_suggestion": None
            }

        if not results_map:
            raise RuntimeError("All scene generations failed. No valid image was produced for auditing.")

        print("🔗 [Chain] Step 3: Auditing candidate scenes...")
        retry_count = 0
        while retry_count <= self.MAX_AUDIT_RETRIES:
            all_passed = True
            print(f"\n🩺 [Chain] Audition {retry_count} ")

            steps_to_fix = []

            for step_id, data in results_map.items():
                if data["is_passed"]:
                    continue

                audit_res = None
                metrics = {}
                max_api_retries = 2

                for i in range(max_api_retries):
                    try:
                        if not data.get("image_path") or not os.path.exists(data["image_path"]):
                            raise FileNotFoundError(f"Missing image for step {step_id}: {data.get('image_path')}")

                        metrics = self.filter.get_physical_report(data["image_path"]) or {}
                        audit_res = self.therapist.audit_scene(
                            data["image_path"],
                            data["scene_data"],
                            metrics,
                            user_input,
                            clinical_insight
                        )
                        if audit_res:
                            break
                    except Exception as api_err:
                        print(f"⚠️ API Attempt {i + 1} failed: {api_err}. Retrying...")
                        await asyncio.sleep(2)

                if isinstance(audit_res, list):
                    audit_res = audit_res[0] if len(audit_res) > 0 else {}

                if not isinstance(audit_res, dict):
                    audit_res = {
                        "decision": "FAIL",
                        "clinical_critique": "Model Output Error",
                        "refinement_suggestion": "Regenerate a safer, clearer restorative natural scene."
                    }

                results_map[step_id]["audit_report"] = audit_res
                results_map[step_id]["metrics"] = metrics
                results_map[step_id]["last_suggestion"] = audit_res.get(
                    "refinement_suggestion", "No suggestion"
                )

                session_logs["iteration_history"].append({
                    "type": "audit",
                    "round": retry_count,
                    "step": step_id,
                    "prompt": data["scene_data"]['image_prompt'],
                    "image_path": data["image_path"],
                    "metrics": metrics,
                    "audit_decision": audit_res.get("decision"),
                    "clinical_critique": audit_res.get("clinical_critique"),
                    "refinement_suggestion": audit_res.get("refinement_suggestion")
                })

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
                if isinstance(refined_plan, list) and len(refined_plan) > 0:
                    refined_plan = refined_plan[0]

                if not isinstance(refined_plan, dict) or 'scenes' not in refined_plan:
                    print(f"⚠️ RAG Refine Format Error for Step {step_id}, using fallback.")
                    refined_plan = {'scenes': [old_data["scene_data"]]}

                current_suggestion = old_data["last_suggestion"] or "Corrective iteration"
                new_scene = refined_plan['scenes'][0]
                new_prompt = new_scene['image_prompt']

                _, new_img_path = await self.production.generate_image(
                    new_prompt,
                    step_id,
                    folder=path_audit,
                    filename=f"step_{step_id}_retry_{retry_count}.jpg",
                    ref_image_path=new_scene.get("reference_image_path"),
                    scene_data=new_scene
                )

                if new_img_path and os.path.exists(new_img_path):
                    results_map[step_id].update({
                        "image_path": new_img_path,
                        "scene_data": new_scene,
                        "reference_image_path": new_scene.get("reference_image_path"),
                        "reference_filename": new_scene.get("reference_filename"),
                        "is_passed": False
                    })

                    session_logs["iteration_history"].append({
                        "type": "refine",
                        "round": retry_count,
                        "step": step_id,
                        "new_prompt": new_prompt,
                        "based_on_suggestion": current_suggestion,
                        "new_image_path": new_img_path,
                        "audit_decision": "PENDING"
                    })
                else:
                    print(f"⚠️ Refinement generation failed for Step {step_id}; keeping previous image.")

        print("\n✅ Audit finished. Saving final images...")

        final_scenes = []
        for step_id in sorted(results_map.keys()):
            data = results_map[step_id]
            source_path = data["image_path"]

            final_filename = f"scene_{step_id}.png"
            final_path = os.path.join(path_case_final, final_filename)
            supir_input_path = os.path.join(path_supir_input, final_filename)

            try:
                shutil.copy2(source_path, final_path)
                shutil.copy2(source_path, supir_input_path)
                print(f"✅ Step {step_id}: saved final image and SUPIR input -> {supir_input_path}")
            except Exception as e:
                print(f"⚠️ Final image copy error for Step {step_id}: {e}")
                final_path = source_path

            metrics = data.get("metrics") or {}
            audit_report = data.get("audit_report") or {}
            scene_data = data.get("scene_data") or {}

            final_scenes.append({
                "step": step_id,
                "image_path": final_path,
                "image_url": f"http://localhost:8000/{final_path.replace(os.sep, '/')}",
                "duration": scene_data.get("duration", 60),
                "audit_critique": audit_report.get("clinical_critique", "Final version"),
                "is_final_passed": data.get("is_passed", False),
                "unity_config": {
                    "kelvin": metrics.get("estimated_kelvin", 4000),
                    "intensity": scene_data.get("intensity", 1.0)
                }
            })

        final_json = {
            "user_input": user_input,
            "clinical_strategy": clinical_insight.get('search_query'),
            "intervention_plan": final_scenes,
            "music_playlist": music_resources,
            "status": "Success",
            "model_used": self.model_name,
            "total_audit_rounds": retry_count,
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
        shared_filt = ToolBox(hf_token=os.getenv("HF_TOKEN"))
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
    get_shared_agents()
    print("✅ Pipeline components loaded. Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=65)
