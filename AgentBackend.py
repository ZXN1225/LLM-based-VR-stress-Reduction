"""Session-scoped tool implementations reusing the original RAG/generation/audit modules."""
import asyncio
import json
import os
import shutil
import time
from copy import deepcopy
from pathlib import Path
from PipelineGuardrails import (CrossProcessAsyncLock, append_session_record, create_session_layout,
    detect_risk_flags, new_seed, public_static_url, record_critique, remember_candidate,
    risk_blocked_result, select_best_candidate)
from PlanSchemas import InterventionPlanSpec, RefinementPlanSpec

GPU_LOCK = CrossProcessAsyncLock()
_shared = None
FALLBACK_AUDIO = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"


async def gpu_work(fn, *args, **kwargs):
    """Do not release GPU ownership on cancellation while a worker is still using the pipeline."""
    async with GPU_LOCK:
        task = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await task
            finally:
                raise


def shared_components():
    global _shared
    if _shared is None:
        from Production_Agent import ProductionAgent
        from MetricsToolBox import ToolBox
        _shared = (ProductionAgent(os.getenv("SUNO_API_KEY"), os.getenv("SUNO_API_BASE")),
                   ToolBox(hf_token=os.getenv("HF_TOKEN")))
    return _shared


def api_key(model):
    lower = model.lower()
    for token, variable in [("gpt", "OPENAI_API_KEY"), ("claude", "CLAUDE_API_KEY"),
                            ("gemini", "GEMINI_API_KEY"), ("deepseek", "DEEPSEEK_API_KEY")]:
        if token in lower:
            return os.getenv(variable) or os.getenv("API_KEY")
    return os.getenv("API_KEY")


class AgentBackend:
    def __init__(self, mode, user_input, context, model, rag=None, auditor=None, components=None):
        self.mode = mode
        self.user_input, self.context, self.model = user_input, deepcopy(context), model
        self.psych = self.context.get("psych_state", {})
        self.session_id, self.paths = create_session_layout(mode)
        self.rag, self.auditor, self.components = rag, auditor, components
        self.references = None
        self.insight = None
        self.plan = None
        self.scenes = {}
        self.music_tasks = []
        self.result = None
        self.max_retries = max(0, int(os.getenv("MAX_AUDIT_RETRIES", "3")))
        self.logs = {"session_id": self.session_id, "start_time": time.time(),
                     "planning_context": self.context, "iteration_history": [], "audit_retries": 0}
        flags = detect_risk_flags(user_input, self.psych)
        if flags:
            self.result = risk_blocked_result(self.session_id, self.export_mode, user_input, self.psych, flags)

    @property
    def export_mode(self):
        return "dialog_guided_personalization" if self.mode == "interactive" else "single_prompt_auto_extraction"

    def ensure_llm_components(self):
        if self.rag is None:
            from RAG_Agent import RAGAgent
            self.rag = RAGAgent(api_key(self.model), os.getenv("CHROMA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"), model_name=self.model)
        if self.auditor is None:
            from Auditing_Agent import AuditingAgent
            self.auditor = AuditingAgent(api_key(self.model), os.getenv("CHROMA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"), model_name=self.model)

    def summary(self):
        return {"session_id": self.session_id, "mode": self.mode,
                "retrieved": self.references is not None, "planned": self.plan is not None,
                "music_started": bool(self.music_tasks), "max_scene_retries": self.max_retries,
                "finished": self.result is not None,
                "scenes": [{"step": k, "status": v["status"], "retries": v["retries"],
                            "error": v.get("error", "")[:250],
                            "critique": (v.get("audit_report") or {}).get("clinical_critique", "")[:400]}
                           for k, v in sorted(self.scenes.items())]}

    async def dispatch(self, name, args):
        if self.result is not None:
            raise ValueError("Session is blocked or already finalized")
        if name == "inspect_session":
            return
        await getattr(self, "tool_" + name)(**args)

    def require_plan(self):
        if self.plan is None:
            raise ValueError("Plan the session first")

    def select(self, steps, statuses):
        self.require_plan()
        if any(k not in self.scenes or self.scenes[k]["status"] not in statuses for k in steps):
            raise ValueError("Every selected step must have eligible status: " + ", ".join(statuses))
        return [self.scenes[k] for k in steps]

    async def tool_retrieve_context(self):
        if self.references is not None:
            return  # session-local memoization; never shares user retrieval results
        def retrieve():
            self.ensure_llm_components()
            self.insight = self.rag._clinical_reasoning_from_state(self.context)
            query = (f"Dialog psychological state: {self.context.get('dialog_summary', '')}. "
                     f"Therapeutic target: {self.insight.get('search_query', '')}")
            return self.rag._query_references(query, n_results=5,
                metadata_filter=self.rag._filter_from_clinical_insight(self.insight))
        self.references = await asyncio.to_thread(retrieve)
        self.logs["clinical_insight"] = self.insight

    async def tool_plan_session(self):
        if self.references is None:
            raise ValueError("Retrieve context first")
        if self.plan is not None:
            return
        plan = await asyncio.to_thread(self.rag._generate_plan,
            json.dumps(self.context, ensure_ascii=False), self.insight, self.references)
        self.plan = InterventionPlanSpec.model_validate(plan).model_dump()
        self.scenes = {s["step"]: {"scene_data": s, "status": "planned", "retries": 0,
                        "seed": new_seed(), "best_candidate": None, "critique_history": [],
                        "reference_image_path": s.get("reference_image_path"),
                        "reference_filename": s.get("reference_filename")}
                       for s in self.plan["scenes"]}

    async def _components(self):
        if self.components is None:
            self.components = await gpu_work(shared_components)
        return self.components

    async def tool_start_music(self):
        self.require_plan()
        if self.music_tasks:
            return
        production, _ = await self._components()
        self.music_tasks = [asyncio.create_task(asyncio.to_thread(production.generate_music,
            t["music_prompt"], t["style"], t["title"])) for t in self.plan["music_playlist"]]
        await asyncio.sleep(0)  # schedule workers before returning to GPU work

    async def _generate(self, selected):
        production, _ = await self._components()
        def work():
            items = []
            for data in selected:
                scene = data["scene_data"]
                try:
                    result = asyncio.run(production.generate_image(scene["image_prompt"], scene["step"],
                        folder=self.paths["base"] if data["retries"] == 0 else self.paths["audit"],
                        filename=f"step_{scene['step']}_round_{data['retries']}.jpg",
                        ref_image_path=scene.get("reference_image_path"), scene_data=scene, seed=data["seed"]))
                    path = result[1] if result else None
                    if not path or not os.path.exists(path):
                        raise RuntimeError("No image was produced")
                    data["base_image_path"] = path
                    data["image_path"] = path
                    items.append({"step": scene["step"], "image_path": path, "prompt": scene["image_prompt"]})
                    data["status"] = "generated"
                except Exception as exc:
                    data["error"] = str(exc)
                    # Failed retry retains the previous candidate but consumes its retry budget.
                    data["status"] = "failed" if data.get("best_candidate") else "generation_failed"
            try:
                repaired = production.refine_seams_for_paths(items,
                    output_folder=self.paths["seam_retry"], overwrite=False) if items else {}
            except Exception as exc:
                self.logs.setdefault("warnings", []).append("Seam repair fallback: " + str(exc))
                repaired = {}
            for data in selected:
                if data["status"] == "generated":
                    candidate = (repaired or {}).get(data["scene_data"]["step"])
                    if candidate and os.path.exists(candidate):
                        data["image_path"] = candidate
                self.logs["iteration_history"].append({"type": "generation", "step": data["scene_data"]["step"],
                    "round": data["retries"], "seed": data["seed"], "status": data["status"],
                    "prompt": data["scene_data"]["image_prompt"], "image_path": data.get("image_path"),
                    "reference_filename": data.get("reference_filename")})
        await gpu_work(work)

    async def tool_generate_scenes(self, steps):
        await self._generate(self.select(steps, {"planned"}))

    async def tool_audit_scenes(self, steps):
        selected = self.select(steps, {"generated"})
        _, metrics_tool = await self._components()
        for data in selected:
            report, metrics = None, {}
            for attempt in range(2):
                try:
                    metrics = await gpu_work(metrics_tool.get_physical_report, data["image_path"]) or {}
                    report = await asyncio.to_thread(self.auditor.audit_scene, data["image_path"],
                        data["scene_data"], metrics, self.user_input, self.insight, psych_state=self.psych)
                    if not isinstance(report, dict) or report.get("decision") not in {"PASS", "FAIL"}:
                        raise ValueError("Invalid audit response")
                    break
                except Exception as exc:
                    report = {"decision": "FAIL", "clinical_critique": "Audit error: " + str(exc),
                              "refinement_suggestion": "Retry with original prompt"}
                    if attempt == 0:
                        await asyncio.sleep(2)
            data["audit_report"], data["metrics"] = report, metrics
            data["is_stagnated"] = record_critique(data, report.get("clinical_critique", ""))
            remember_candidate(data, report, metrics, data["retries"])
            data["is_passed"] = report["decision"] == "PASS"
            data["status"] = "passed" if data["is_passed"] else "failed"
            self.logs["iteration_history"].append({"type": "audit", "step": data["scene_data"]["step"],
                "round": data["retries"], "metrics": metrics, "audit": report, "seed": data["seed"]})

    async def tool_refine_scenes(self, steps):
        selected = self.select(steps, {"failed"})
        if any(d["retries"] >= self.max_retries for d in selected):
            raise ValueError("A selected scene has exhausted its retry budget")
        revisions = []
        for data in selected:
            feedback = deepcopy(data["audit_report"])
            stagnated = data.get("is_stagnated", False)
            if stagnated:
                feedback["refinement_suggestion"] = feedback.get("refinement_suggestion", "") + " Change composition/reference while preserving the original goal."
            plan = await asyncio.to_thread(self.rag.refine_intervention_plan, data["scene_data"], feedback,
                self.context.get("dialog_summary", self.user_input), original_insight=self.insight,
                psych_state=self.psych, exclude_reference_filename=data.get("reference_filename") if stagnated else None)
            scene = RefinementPlanSpec.model_validate(plan).model_dump()["scenes"][0]
            if scene["step"] != data["scene_data"]["step"]:
                raise ValueError("Refinement must preserve the selected scene step")
            revisions.append(scene)
        # Commit only after every requested refinement has validated successfully.
        for data, scene in zip(selected, revisions):
            stagnated = data.get("is_stagnated", False)
            data["scene_data"] = scene
            data["reference_image_path"] = data["scene_data"].get("reference_image_path")
            data["reference_filename"] = data["scene_data"].get("reference_filename")
            data["seed"] = new_seed(data["seed"], force_new_segment=stagnated)
            data["retries"] += 1
        self.logs["audit_retries"] = max(d["retries"] for d in self.scenes.values())
        await self._generate(selected)

    async def tool_finish_session(self):
        self.require_plan()
        if not self.music_tasks:
            raise ValueError("Start music before finalization")
        for data in self.scenes.values():
            if data["status"] not in {"passed", "generation_failed"}:
                if data["status"] != "failed" or data["retries"] < self.max_retries:
                    raise ValueError("All scenes must be audited and terminal before finalization")
        if not any(d.get("best_candidate") for d in self.scenes.values()):
            raise RuntimeError("All initial scenes failed; no successful session can be exported")
        production, _ = await self._components()
        final_scenes = []
        for step, data in sorted(self.scenes.items()):
            if not data.get("best_candidate"):
                continue
            select_best_candidate(data)
            filename = f"scene_{step}{Path(data['image_path']).suffix.lower() or '.png'}"
            final_path = os.path.join(self.paths["final"], filename)
            upscale_input = os.path.join(self.paths["upscale_input"], filename)
            upscale_output = os.path.join(self.paths["upscale_output"], f"scene_{step}.png")
            shutil.copy2(data["image_path"], final_path)
            shutil.copy2(data["image_path"], upscale_input)
            s, m, a = data["scene_data"], data["metrics"], data["audit_report"]
            final_scenes.append({"step": step, "image_path": final_path, "image_url": public_static_url(final_path),
                "upscale_input_path": upscale_input, "upscale_output_path": upscale_output, "upscale_enhanced": False,
                "duration": s.get("duration", 60), "audit_critique": a.get("clinical_critique", ""),
                "is_final_passed": data["is_passed"], "seed": data["seed"], "selected_candidate_round": data.get("selected_candidate_round"),
                "therapeutic_goal": s["therapeutic_goal"], "target_kelvin": s["target_kelvin"], "quality_metrics": m,
                "unity_config": {"kelvin": m.get("estimated_kelvin", s["target_kelvin"]), "intensity": s["intensity"]}})
        if os.getenv("AUTO_RUN_UPSCALE", "1") == "1":
            try:
                enhanced = await gpu_work(production.upscale_images_batch,
                    [s["upscale_input_path"] for s in final_scenes], [s["upscale_output_path"] for s in final_scenes],
                    scale=int(os.getenv("AUTO_UPSCALE_SCALE", os.getenv("PHOTO_FINISHER_SCALE", "4"))))
                for s in final_scenes:
                    path = (enhanced or {}).get(s["upscale_output_path"])
                    if path and os.path.exists(path):
                        s.update(image_path=path, image_url=public_static_url(path), upscale_enhanced=True)
            except Exception as exc:
                self.logs.setdefault("warnings", []).append("Final enhancement fallback: " + str(exc))
        music = await asyncio.gather(*self.music_tasks, return_exceptions=True)
        playlist = [{"step": t["step"], "title": t["title"],
                     "audio_url": url if isinstance(url, str) and url else FALLBACK_AUDIO}
                    for t, url in zip(self.plan["music_playlist"], music)]
        complete = len(final_scenes) == 10 and all(s["is_final_passed"] for s in final_scenes)
        self.result = {"session_id": self.session_id, "user_input": self.user_input, "psych_state": self.psych,
            "clinical_strategy": self.insight.get("search_query"), "clinical_insight": self.insight,
            "intervention_plan": final_scenes, "music_playlist": playlist,
            "status": "Success" if complete else "Partial", "model_used": self.model, "mode": self.export_mode,
            "total_audit_rounds": self.logs["audit_retries"], "planned_scene_count": 10,
            "missing_steps": sorted(set(range(1, 11)) - {s["step"] for s in final_scenes}),
            "metrics_stage": "before_final_photo_enhancement", "orchestration": "function_calling"}

    async def close(self):
        # Thread-backed requests cannot safely be killed by Task.cancel(). Drain bounded HTTP work.
        if self.music_tasks:
            await asyncio.gather(*self.music_tasks, return_exceptions=True)

    def record_agent(self, trace, usage):
        self.logs.update(tool_trace=trace, coordinator_usage=usage, end_time=time.time())
        self.logs["duration_seconds"] = self.logs["end_time"] - self.logs["start_time"]
        result = self.result or {"session_id": self.session_id, "mode": self.export_mode,
                                 "status": "failed", "intervention_plan": []}
        Path(self.paths["root"], "session.json").write_text(json.dumps({"result": result, "logs": self.logs},
            ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        append_session_record(result, self.logs)
