"""One local FastAPI/Gradio application for Interactive and Passive input modes."""
import asyncio
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
load_dotenv()  # Load GPU_LOCK_FILE before AgentBackend creates its shared lock.
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import gradio as gr
from StateManager import StateManager
from PipelineGuardrails import detect_risk_flags
from AgentBackend import AgentBackend, api_key
from ToolAgent import ToolAgent

MODEL = os.getenv("LLM_MODEL", "gpt-5.4")
COORDINATOR_MODEL = os.getenv("AGENT_MODEL")
MODEL_CHOICES = list(dict.fromkeys([MODEL] + [x.strip() for x in os.getenv("LLM_MODELS",
    "gpt-5.4,gpt-5.4-mini,claude-opus-4.7,gemini-3.1-pro,deepseek-v4").split(",") if x.strip()]))
Path("static").mkdir(exist_ok=True)
app = FastAPI(title="VR Tool-Calling Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")
results = OrderedDict()


def dialog_agent(model=MODEL):
    from DialogTherapistAgent import DialogTherapistAgent
    return DialogTherapistAgent(api_key(model), model_name=model, max_turns=5)


async def run_session(mode, description, dialog_state=None, model_name=MODEL):
    description = description.strip()
    if model_name not in MODEL_CHOICES:
        raise ValueError("所选模型未配置。请检查 LLM_MODELS。")
    if mode not in {"passive", "interactive"}:
        raise ValueError("请选择有效的输入方式。")
    if len(description) > 12000:
        raise ValueError("单次需求请控制在 12000 个字符以内。")
    if mode == "interactive":
        state = StateManager.ensure_state(dialog_state)
        # Include every user turn in the pre-generation deterministic risk gate.
        description = "\n".join([description] + [m["content"] for m in state["history"] if m["role"] == "user"])
        flags = detect_risk_flags(description, state["psych_state"])
        if not flags and not StateManager.can_generate(state):
            raise ValueError("请继续对话，或选择使用当前信息。")
        context = dialog_agent(model_name).build_planning_context(state)
        if flags:
            context["psych_state"]["risk_flags"] = flags
    else:
        if not description:
            raise ValueError("请先输入你的需求。")
        flags = detect_risk_flags(description, {})
        if flags:
            context = {"psych_state": {"risk_flags": flags}, "original_user_need": description}
        else:
            from PassiveState import SinglePromptPsychStateExtractor, build_passive_context
            extractor = SinglePromptPsychStateExtractor(api_key(model_name), model_name=model_name)
            psych, metadata = await asyncio.to_thread(extractor.extract, description)
            context = build_passive_context(description, psych, metadata["confidence"])
            context["state_extraction"] = metadata
    backend = AgentBackend(mode, description, context, model_name)
    if mode == "interactive":
        backend.logs["dialog_state"] = state
    if backend.result is not None:
        backend.record_agent([], [])
        return backend.result
    coordinator_model = COORDINATOR_MODEL or model_name
    agent = ToolAgent(backend, coordinator_model, api_key(coordinator_model),
                      max_turns=int(os.getenv("AGENT_MAX_TURNS", "64")),
                      max_calls=int(os.getenv("AGENT_MAX_CALLS", "96")))
    result = await agent.run()
    if mode == "interactive":
        result["dialog_state"] = dialog_state
    results[result["session_id"]] = result
    while len(results) > 32:
        results.popitem(last=False)
    return result


class SessionRequest(BaseModel):
    mode: Literal["passive", "interactive"] = "passive"
    description: str = Field(default="", max_length=12000)
    dialog_state: dict | None = None
    model_name: str = MODEL


@app.post("/generate-session")
async def create_session(request: SessionRequest):
    try:
        result = await run_session(request.mode, request.description, request.dialog_state, request.model_name)
        return {"session_id": result["session_id"], "data": result}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, "生成未完成，请查看本机会话日志。") from exc


@app.get("/get-latest-session")
async def get_session(session_id: str):
    if session_id not in results:
        raise HTTPException(404, "Session not retained in memory; inspect its session.json locally")
    return results[session_id]


def state_label(state):
    if StateManager.has_risk(state):
        return "当前信息触发风险拦截，无法继续自动生成。"
    return "信息已准备好，可以生成。" if StateManager.can_generate(state) else "可以继续补充环境与光照偏好。"


async def chat(message, state, model_name):
    state = StateManager.ensure_state(state)
    if len(message) > 12000:
        return StateManager.chatbot_messages(state), state, "单条消息请控制在 12000 个字符以内。", message
    if message.strip():
        if model_name not in MODEL_CHOICES:
            raise ValueError("模型未配置。")
        _, state = await asyncio.to_thread(dialog_agent(model_name).step, message.strip(), state)
    return StateManager.chatbot_messages(state), state, state_label(state), ""


def finalize(state):
    state = StateManager.finalize(state)
    return state, state_label(state)


def switch_mode(mode):
    # Drop hidden dialogue and prior output whenever input mode changes.
    return (gr.update(visible=mode == "interactive"), gr.update(visible=mode == "passive"),
            StateManager.new_state(), [], "", "", [], None,
            gr.update(choices=[], value=None), "", "")


async def generate(mode, prompt, state, model_name):
    try:
        result = await run_session(mode, prompt if mode == "passive" else "", state, model_name)
        gallery = [(s["image_path"], f"场景 {s['step']}") for s in result.get("intervention_plan", [])]
        urls = [t["audio_url"] for t in result.get("music_playlist", [])]
        selected = urls[0] if urls else None
        message = {"Success": "生成完成。", "Partial": "生成结束，部分场景未通过审核或生成失败，请查看结果详情。",
                   "blocked_risk": result.get("crisis_guidance", "生成已被风险规则拦截。")}.get(result["status"], "生成结束。")
        return gallery, selected, gr.update(choices=urls, value=selected), json.dumps(result, ensure_ascii=False, indent=2), message
    except Exception as exc:
        return [], None, gr.update(choices=[], value=None), "", "生成未完成：" + str(exc)[:500]


with gr.Blocks(title="VR 场景生成助手") as demo:
    gr.Markdown("# VR 场景生成助手\n选择输入方式，描述你希望体验的自然环境。")
    mode = gr.Radio(choices=[("直接描述（Passive）", "passive"), ("对话引导（Interactive）", "interactive")],
                    value="passive", label="输入方式")
    with gr.Accordion("模型设置", open=False):
        model_selector = gr.Dropdown(choices=MODEL_CHOICES, value=MODEL, label="模型", interactive=True)
    state = gr.State(StateManager.new_state())
    with gr.Group(visible=True) as passive_group:
        prompt = gr.Textbox(label="你的需求", lines=4,
                            placeholder="例如：最近有些疲惫，想看开阔、温暖的湖边，不喜欢人群。")
    with gr.Group(visible=False) as interactive_group:
        chatbot = gr.Chatbot(label="对话", height=300)
        message = gr.Textbox(label="补充你的想法")
        with gr.Row():
            send = gr.Button("发送")
            finish = gr.Button("使用当前信息")
    status = gr.Markdown()
    generate_button = gr.Button("生成场景与音乐", variant="primary")
    gallery = gr.Gallery(label="场景", columns=2, object_fit="contain")
    audio = gr.Audio(label="音乐", interactive=False)
    selector = gr.Dropdown(label="切换音乐", choices=[], interactive=True)
    with gr.Accordion("结果与对接数据", open=False):
        result_json = gr.Code(language="json", label="结果")
    reset = gr.Button("清空")
    # Serialize UI mutations so a pending generation cannot race a mode reset.
    ui_queue = {"concurrency_id": "ui-events", "concurrency_limit": 1}
    send.click(chat, [message, state, model_selector], [chatbot, state, status, message], **ui_queue)
    message.submit(chat, [message, state, model_selector], [chatbot, state, status, message], **ui_queue)
    finish.click(finalize, [state], [state, status], **ui_queue)
    generate_button.click(generate, [mode, prompt, state, model_selector], [gallery, audio, selector, result_json, status], **ui_queue)
    selector.change(lambda x: x, selector, audio)
    reset_outputs = [interactive_group, passive_group, state, chatbot, prompt, message,
                     gallery, audio, selector, result_json, status]
    mode.change(switch_mode, mode, reset_outputs, **ui_queue)
    model_selector.change(switch_mode, mode, reset_outputs, **ui_queue)
    reset.click(switch_mode, mode, reset_outputs, **ui_queue)

demo.queue(default_concurrency_limit=1)
app = gr.mount_gradio_app(app, demo, path="/gui")


def main():
    import uvicorn
    uvicorn.run(app, host=os.getenv("APP_HOST", "127.0.0.1"), port=int(os.getenv("APP_PORT", "8000")))


if __name__ == "__main__":
    main()
