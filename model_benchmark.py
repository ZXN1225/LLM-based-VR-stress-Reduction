import asyncio
import pandas as pd
import json
import os
import time
from AgentAPP import ProductionAgent, FilterAgent,TherapyChain

TEST_MODELS = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "gemini/gemini-3.1-pro-preview",
    "deepseek/deepseek-v4-pro",
    "anthropic/claude-opus-4-7"
]


async def run_benchmark():

    try:
        df = pd.read_csv("User input - Cognitive Fatigue.csv")
        test_cases = df['User Text'].tolist()[:8]
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    results = []
    p_agent = ProductionAgent(
        suno_api_key=os.getenv("SUNO_API_KEY"),
        suno_base_url=os.getenv("SUNO_API_BASE"))
    f_agent = FilterAgent(hf_token=os.getenv("HF_TOKEN"))

    for model in TEST_MODELS:
        print(f"\n🚀 Testing Model: {model}")
        try:
            chain = TherapyChain(model_name=model, shared_production=p_agent, shared_filter=f_agent)
        except Exception as e:
            print(f"⚠️ Failed to initialize chain for {model}: {e}")
            continue

        for idx, user_input in enumerate(test_cases):
            print(f"  📝 Processing Case {idx + 1}/{len(test_cases)}...")

            start_time = time.time()
            try:
                final_data, logs = await chain.execute(user_input, case_id=idx)
                latency = time.time() - start_time
                img_path = None
                if final_data.get('intervention_plan'):
                    img_path = final_data['intervention_plan'][0].get('image_path')

                history = logs.get('iteration_history', [])
                final_prompt = history[-1].get('prompt') if history else "N/A"
                final_metrics = history[-1].get('metrics') if history else {}
                metrics_evo = [h.get('metrics') for h in history if h.get('metrics') is not None]

                results.append({
                    "model": model,
                    "case_id": idx,
                    "user_input": user_input,
                    "latency_seconds": round(latency, 2),
                    "audit_retries": logs.get('audit_retries', 0),
                    "clinical_insight": json.dumps(logs.get('clinical_insight', {}), ensure_ascii=False),
                    "final_prompt": final_prompt,
                    "final_metrics": json.dumps(final_metrics, ensure_ascii=False),
                    "metrics_evolution": json.dumps(metrics_evo, ensure_ascii=False),
                    "full_iteration_log": json.dumps(history, ensure_ascii=False),
                    "image_path": img_path
                })
                print(f"  ✅ Case {idx + 1} Done. (Retries: {logs.get('audit_retries')}, Latency: {round(latency, 2)}s)")

            except Exception as e:
                print(f"  ❌ Error in {model} on Case {idx}: {e}")
                results.append({
                    "model": model,
                    "case_id": idx,
                    "user_input": user_input,
                    "error": str(e),
                    "final_decision": "ERROR"
                })

            await asyncio.sleep(1.5)

    output_path = f"benchmark_results_{int(time.time())}.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 30)
    print(f"✅ Benchmark Complete!")
    print(f"📊 Results saved to: {output_path}")
    print(f"🖼️ Images saved in: static/results/[model_name]/case_[id]/")
    print("=" * 30)


if __name__ == "__main__":
    asyncio.run(run_benchmark())