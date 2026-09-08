"""Offline research evaluation for persisted session_logs.jsonl records."""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _mean(values):
    return statistics.fmean(values) if values else None


def _cohen_kappa(pairs):
    if not pairs:
        return None
    labels = sorted({label for pair in pairs for label in pair})
    observed = sum(a == b for a, b in pairs) / len(pairs)
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum((left[label] / len(pairs)) * (right[label] / len(pairs)) for label in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def audit_kappa(csv_path):
    if not csv_path:
        return None
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        pairs = [
            (row["human_decision"].strip().upper(), row["model_decision"].strip().upper())
            for row in rows
            if row.get("human_decision") and row.get("model_decision")
        ]
    return {"labeled_items": len(pairs), "cohen_kappa": _cohen_kappa(pairs)}


def retrieval_recall(jsonl_path):
    if not jsonl_path:
        return None
    recalls = []
    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            expected = set(row.get("expected_references", []))
            retrieved = set(row.get("retrieved_references", []))
            if expected:
                recalls.append(len(expected & retrieved) / len(expected))
    return {"queries": len(recalls), "mean_recall": _mean(recalls)}


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        result = record.get("result", {})
        groups[result.get("mode", "unknown")].append(record)

    output = {}
    for mode, rows in groups.items():
        eligible = [row for row in rows if row.get("result", {}).get("status") != "blocked_risk"]
        all_scene_pass = []
        ds_scores, md_scores, kelvin_matches = [], [], []
        retries = []
        for row in eligible:
            result, logs = row.get("result", {}), row.get("logs", {})
            scenes = result.get("intervention_plan", [])
            all_scene_pass.append(bool(scenes) and all(scene.get("is_final_passed") for scene in scenes))
            retries.append(int(result.get("total_audit_rounds", logs.get("audit_retries", 0)) or 0))
            for scene in scenes:
                metrics = scene.get("quality_metrics", {}) or {}
                if isinstance(metrics.get("ds_score"), (int, float)):
                    ds_scores.append(float(metrics["ds_score"]))
                if isinstance(metrics.get("md_score"), (int, float)):
                    md_scores.append(float(metrics["md_score"]))
                actual = (scene.get("unity_config") or {}).get("kelvin")
                target = scene.get("target_kelvin")
                if isinstance(actual, (int, float)) and isinstance(target, (int, float)):
                    kelvin_matches.append(abs(actual - target) <= 800)

        output[mode] = {
            "sessions_total": len(rows),
            "sessions_risk_blocked": len(rows) - len(eligible),
            "eligible_sessions": len(eligible),
            "pass_within_retry_budget_rate": _mean(all_scene_pass),
            "mean_audit_retries": _mean(retries),
            "mean_ds_score": _mean(ds_scores),
            "mean_md_score": _mean(md_scores),
            "kelvin_within_800k_rate": _mean(kelvin_matches),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_jsonl")
    parser.add_argument("--audit-labels", help="CSV with human_decision and model_decision columns")
    parser.add_argument("--retrieval-golden", help="JSONL with expected_references and retrieved_references arrays")
    args = parser.parse_args()

    path = Path(args.session_jsonl)
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    report = {
        "condition_summary": summarize(records),
        "audit_reliability": audit_kappa(args.audit_labels),
        "retrieval_recall": retrieval_recall(args.retrieval_golden),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
