"""
FleetAI - Evaluation Harness

Compare baseline (heuristic) vs SFT vs GRPO trained models.

Usage:
  # Baseline only (no API needed)
  python evaluate.py --mode baseline

  # Evaluate a trained model via API
  API_KEY=your-key MODEL_NAME=gpt-4o-mini python evaluate.py --mode trained

  # Full comparison table
  API_KEY=your-key MODEL_NAME=gpt-4o-mini python evaluate.py --mode compare
"""

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Optional

from utils import http_post, http_get, call_llm, build_inspector_prompt, parse_json_from_response, extract_inspector_action, build_heuristic_action

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "")

TASKS = ["inspection_easy", "inspection_hard", "inspection_adversarial"]
TASK_LABELS = {
    "inspection_easy": "Easy (Obvious Errors)",
    "inspection_hard": "Hard (Subtle Errors + Clean Traps)",
    "inspection_adversarial": "Adversarial (Designed to Trick)",
}


def evaluate_mode(mode: str, num_tickets: int, seed: int) -> Dict[str, Any]:
    results = {}
    for task in TASKS:
        scores = []
        component_totals = {}
        details = []

        for i in range(num_tickets):
            try:
                reset_result = http_post(f"{ENV_URL}/reset", {"task": task, "seed": seed + i})
                observation = reset_result.get("observation", reset_result)
            except Exception as e:
                scores.append(0.0)
                details.append({"error": str(e)})
                continue

            action = None

            if mode == "trained":
                prompt = build_inspector_prompt(observation, task)
                api_key = API_KEY or OPENAI_API_KEY
                response_text = call_llm(prompt, MODEL_NAME, api_key, API_BASE_URL)
                if response_text:
                    parsed = parse_json_from_response(response_text)
                    if parsed:
                        action = extract_inspector_action(parsed)

            if action is None:
                action = build_heuristic_action(observation)

            try:
                step_result = http_post(f"{ENV_URL}/step", action)
                score = step_result.get("reward", 0.0)
                scores.append(score)

                grader = step_result.get("info", {}).get("grader", {})
                for comp in grader.get("components", []):
                    name = comp.get("name", "")
                    if name not in component_totals:
                        component_totals[name] = []
                    component_totals[name].append(comp.get("score", 0.0))

                details.append({
                    "ticket_id": observation.get("ticket", {}).get("id"),
                    "score": score,
                    "flagged": action.get("flagged"),
                })
            except Exception as e:
                scores.append(0.0)
                details.append({"error": str(e)})

        mean = sum(scores) / len(scores) if scores else 0.0
        component_means = {
            name: round(sum(vals) / len(vals), 4)
            for name, vals in component_totals.items()
        }

        results[task] = {
            "mean": round(mean, 4),
            "min": round(min(scores), 4) if scores else 0.0,
            "max": round(max(scores), 4) if scores else 0.0,
            "scores": scores,
            "components": component_means,
            "details": details,
        }

    return results


def print_results(results: Dict[str, Any], label: str):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    for task in TASKS:
        r = results.get(task, {})
        task_label = TASK_LABELS.get(task, task)
        mean = r.get("mean", 0)
        print(f"  {task_label:40s}: {mean:.3f}")
        components = r.get("components", {})
        for comp_name, comp_val in components.items():
            print(f"    {comp_name:38s}: {comp_val:.3f}")
    print()


def print_comparison(baseline: Dict, trained: Dict):
    print(f"\n{'#' * 70}")
    print("  COMPARISON: Baseline vs Trained")
    print(f"{'#' * 70}")
    print(f"  {'Task':<35s} {'Baseline':>10s} {'Trained':>10s} {'Delta':>10s}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 10}")

    b_overall = []
    t_overall = []

    for task in TASKS:
        label = TASK_LABELS.get(task, task)
        b_mean = baseline.get(task, {}).get("mean", 0)
        t_mean = trained.get(task, {}).get("mean", 0)
        delta = t_mean - b_mean
        arrow = "+" if delta > 0 else ""
        print(f"  {label:<35s} {b_mean:>10.3f} {t_mean:>10.3f} {arrow}{delta:>9.3f}")
        b_overall.extend(baseline.get(task, {}).get("scores", []))
        t_overall.extend(trained.get(task, {}).get("scores", []))

    if b_overall and t_overall:
        b_all = sum(b_overall) / len(b_overall)
        t_all = sum(t_overall) / len(t_overall)
        delta = t_all - b_all
        arrow = "+" if delta > 0 else ""
        print(f"\n  {'OVERALL':<35s} {b_all:>10.3f} {t_all:>10.3f} {arrow}{delta:>9.3f}")

    print(f"\n  Component Comparison:")
    print(f"  {'Component':<35s} {'Baseline':>10s} {'Trained':>10s} {'Delta':>10s}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 10}")

    all_components = set()
    for task in TASKS:
        all_components.update(baseline.get(task, {}).get("components", {}).keys())
        all_components.update(trained.get(task, {}).get("components", {}).keys())

    for comp in sorted(all_components):
        b_vals = [baseline.get(t, {}).get("components", {}).get(comp, 0) for t in TASKS]
        t_vals = [trained.get(t, {}).get("components", {}).get(comp, 0) for t in TASKS]
        b_avg = sum(b_vals) / len(b_vals) if b_vals else 0
        t_avg = sum(t_vals) / len(t_vals) if t_vals else 0
        delta = t_avg - b_avg
        arrow = "+" if delta > 0 else ""
        print(f"  {comp:<35s} {b_avg:>10.3f} {t_avg:>10.3f} {arrow}{delta:>9.3f}")

    print(f"{'#' * 70}")


def main():
    parser = argparse.ArgumentParser(description="FleetAI Evaluation Harness")
    parser.add_argument("--mode", choices=["baseline", "trained", "compare"],
                        default="compare", help="Evaluation mode")
    parser.add_argument("--tickets", type=int, default=10, help="Tickets per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("  FLEET AI - EVALUATION HARNESS")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API Key: {'set' if (API_KEY or OPENAI_API_KEY) else 'NOT SET'}")
    print(f"  Tickets per task: {args.tickets}")
    print("=" * 60)

    try:
        health = http_get(f"{ENV_URL}/health")
        print(f"  Server: healthy")
    except Exception as e:
        print(f"  Server not reachable: {e}")
        print("  Start with: python -m uvicorn server.app:app --port 7860")
        return

    baseline_results = {}
    trained_results = {}

    if args.mode in ("baseline", "compare"):
        print("\n  Evaluating baseline (heuristic)...")
        baseline_results = evaluate_mode("baseline", args.tickets, args.seed)
        print_results(baseline_results, "BASELINE (Heuristic Inspector)")

    if args.mode in ("trained", "compare"):
        if not (API_KEY or OPENAI_API_KEY):
            print("\n  WARNING: No API key. Using heuristic fallback for 'trained' mode.")
        print(f"\n  Evaluating trained ({MODEL_NAME})...")
        trained_results = evaluate_mode("trained", args.tickets, args.seed)
        print_results(trained_results, f"TRAINED ({MODEL_NAME})")

    if args.mode == "compare":
        print_comparison(baseline_results, trained_results)

    output = {
        "mode": args.mode,
        "model": MODEL_NAME,
        "seed": args.seed,
        "tickets_per_task": args.tickets,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": {task: {"mean": r.get("mean", 0), "components": r.get("components", {})}
                     for task, r in baseline_results.items()} if baseline_results else None,
        "trained": {task: {"mean": r.get("mean", 0), "components": r.get("components", {})}
                    for task, r in trained_results.items()} if trained_results else None,
    }

    output_path = "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
