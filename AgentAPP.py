import os
from dotenv import load_dotenv
import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import json


from RAG_Agent import RAGAgent
from Production_Agent import *
from Filter_Agent import FilterAgent

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

load_dotenv()
API_KEY = os.getenv("API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

latest_session_result = None

rag_agent = RAGAgent(API_KEY)
production_agent = ProductionAgent(os.getenv("SUNO_API_KEY"), os.getenv("SUNO_API_BASE"))
filter_agent = FilterAgent(API_KEY, HF_TOKEN)

# Mount static file directory
STATIC_DIRS = [
    "static/base",
    "static/candidate",
    "static/final_images"
]

for folder in STATIC_DIRS:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.mount("/static", StaticFiles(directory="static"), name="static")

class UserRequest(BaseModel):
    description: str

def run_full_intervention_pipeline(user_input: str):
    """Execute full intervention pipeline"""

    # Step1：RAG Strategy Formulation
    plan = rag_agent.get_intervention_plan(user_input)

    # Step 2: Generate matching music
    music_resources = []
    for track in plan.get('music_playlist', []):
        audio_url = production_agent.generate_music(
            track['music_prompt'],
            track['style'],

            track['title']
        )
        music_resources.append({
            "step": track['step'],
            "title": track['title'],
            "audio_url": audio_url
        })

    # Step 3: Production generates candidate scenes
    candidate_scenes = []
    for scene in plan['scenes']:
        img_url, local_path = production_agent.generate_image(scene['image_prompt'], scene['step'])
        candidate_scenes.append({
            "step": scene['step'],
            "duration": scene['duration'],
            "image_url": img_url,
            "image_path": local_path,
            "image_prompt": scene['image_prompt'],
            "unity_config": {
                "kelvin": scene['target_kelvin'],
                "intensity": scene.get('intensity', 1.2)
            }
        })

    # Step 4: Filter to select the final scene(Top10)
    final_scenes = filter_agent.evaluate_scenes(user_input, candidate_scenes)

    return {
        "intervention_plan": final_scenes,
        "music_playlist": music_resources
    }

@app.get("/get-latest-session")
async def get_latest_session():
    """Unity 轮询此接口获取最新生成的场景"""
    global latest_session_result
    return latest_session_result

@app.post("/generate-session")
async def create_session(request: UserRequest):
    """API interfaces for Unity or external calls"""
    try:
        final_data = run_full_intervention_pipeline(request.description)
        return final_data
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def intervention_gui_logic(user_input):
    global latest_session_result
    if not user_input.strip():
        return [], None, gr.update(choices=[]), "Input is empty."

    try:
        result = run_full_intervention_pipeline(user_input)

        latest_session_result = result

        image_list = [s['image_url'] for s in result['intervention_plan']]
        audio_urls = [m['audio_url'] for m in result['music_playlist']]
        initial_audio = audio_urls[0] if audio_urls else None

        return (
            image_list,
            initial_audio,
            gr.update(choices=audio_urls, value=initial_audio),
            json.dumps(result, indent=4)
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
    uvicorn.run(app, host="0.0.0.0", port=8000)