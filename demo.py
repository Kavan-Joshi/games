"""
FleetAI - Before/After Demo for Judges

Shows baseline (heuristic) vs trained (LLM) inspector performance
across all three difficulty levels with detailed per-ticket results.

Usage:
  # Baseline only (no API key needed)
  python demo.py --mode baseline

  # With trained model (requires API key)
  API_KEY=your-key MODEL_NAME=gpt-4o-mini python demo.py --mode trained

  # Full comparison
  API_KEY=your-key MODEL_NAME=gpt-4o-mini python demo.py --mode compare
"""

import os
import sys
import json
import argparse
import time
from typing import Any, Dict, List, Optional

from utils import http_post, http_get, call_llm, build_inspector_prompt, parse_json_from_response, extract_inspector_action, build_heuristic_action

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "")

TICKETS_PER_TASK = 5
SEED = 42

TASKS = ["inspection_easy", "inspection_hard", "inspection_adversarial"]
TASK_LABELS = {
    "inspection_easy": "Easy (Obvious Errors)",
    "inspection_hard": "Hard (Subtle Errors + Clean Traps)",
    "inspection_adversarial": "Adversarial (Designed to Trick)",
}


def run_task(task: str, num_tickets: int, mode: str) -> Dict[str, Any]:
    scores = []
    details = []
    label = TASK_LABELS.get(task, task)
    print(f"\n  [{label}]", flush=True)

    for i in range(num_tickets):
        try:
            reset_result = http_post(f"{ENV_URL}/reset", {"task": task, "seed": SEED + i})
            observation = reset_result.get("observation", reset_result)
            ticket_id = observation.get("ticket", {}).get("id", "unknown")
            has_errors = reset_result.get("info", {}).get("has_injected_errors", False)
            error_fields = reset_result.get("info", {}).get("injected_error_fields", [])
        except Exception as e:
            print(f"    Reset error: {e}", flush=True)
            scores.append(0.0)
            details.append({"ticket_id": "error", "score": 0.0, "error": str(e)})
            continue

        action = None

        if mode in ("trained", "compare"):
            prompt = build_inspector_prompt(observation, task)
            api_key = API_KEY or OPENAI_API_KEY
            response_text = call_llm(prompt, MODEL_NAME, api_key, API_BASE_URL)
            if response_text:
                parsed = parse_json_from_response(response_text)
                if parsed:
                    action = extract_inspector_action(parsed)

        if action is None and mode in ("baseline", "compare"):
            action = build_heuristic_action(observation)

        if action is None:
            action = build_heuristic_action(observation)

        try:
            step_result = http_post(f"{ENV_URL}/step", action)
            score = step_result.get("reward", 0.0)
            grader_info = step_result.get("info", {}).get("grader", {})

            detail = {
                "ticket_id": ticket_id,
                "score": score,
                "had_errors": has_errors,
                "injected_error_fields": error_fields,
                "inspector_flagged": action.get("flagged", False),
                "inspector_flagged_fields": action.get("flagged_fields", []),
                "grader_components": grader_info.get("components", []),
            }
            details.append(detail)
            scores.append(score)
        except Exception as e:
            print(f"    Step error: {e}", flush=True)
            scores.append(0.0)
            details.append({"ticket_id": ticket_id, "score": 0.0, "error": str(e)})

        status = "OK" if score >= 0.5 else "LOW" if score >= 0.2 else "FAIL"
        flagged_str = "FLAGGED" if action.get("flagged") else "CLEAN"
        errors_str = "has-errors" if has_errors else "clean"
        print(f"    {ticket_id}: {score:.3f} [{status}] ({errors_str}, inspector={flagged_str})", flush=True)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "task": task,
        "num_tickets": num_tickets,
        "scores": scores,
        "mean_score": mean_score,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "details": details,
    }


def print_summary(all_results: Dict[str, Any], mode: str):
    print(f"\n{'=' * 60}", flush=True)
    print(f"  {mode.upper()} RESULTS SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)

    overall_scores = []
    for task, result in all_results.items():
        label = TASK_LABELS.get(task, task)
        print(f"  {label:40s}: mean={result['mean_score']:.3f}  "
              f"range=[{result['min_score']:.3f}, {result['max_score']:.3f}]", flush=True)
        overall_scores.extend(result["scores"])

    if overall_scores:
        overall_mean = sum(overall_scores) / len(overall_scores)
        print(f"\n  OVERALL MEAN: {overall_mean:.3f}", flush=True)
    print(f"{'=' * 60}", flush=True)


def print_comparison(baseline_results: Dict, trained_results: Dict):
    print(f"\n{'#' * 60}", flush=True)
    print("  BEFORE vs AFTER COMPARISON", flush=True)
    print(f"{'#' * 60}", flush=True)
    print(f"  {'Task':<40s} {'Baseline':>10s} {'Trained':>10s} {'Delta':>10s}", flush=True)
    print(f"  {'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10}", flush=True)

    baseline_overall = []
    trained_overall = []

    for task in TASKS:
        label = TASK_LABELS.get(task, task)
        b_mean = baseline_results.get(task, {}).get("mean_score", 0.0)
        t_mean = trained_results.get(task, {}).get("mean_score", 0.0)
        delta = t_mean - b_mean
        arrow = "+" if delta > 0 else ""
        print(f"  {label:<40s} {b_mean:>10.3f} {t_mean:>10.3f} {arrow}{delta:>9.3f}", flush=True)
        baseline_overall.extend(baseline_results.get(task, {}).get("scores", []))
        trained_overall.extend(trained_results.get(task, {}).get("scores", []))

    if baseline_overall and trained_overall:
        b_overall = sum(baseline_overall) / len(baseline_overall)
        t_overall = sum(trained_overall) / len(trained_overall)
        delta = t_overall - b_overall
        arrow = "+" if delta > 0 else ""
        print(f"\n  {'OVERALL':<40s} {b_overall:>10.3f} {t_overall:>10.3f} {arrow}{delta:>9.3f}", flush=True)

    print(f"{'#' * 60}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="FleetAI Demo - Before/After Comparison")
    parser.add_argument("--mode", choices=["baseline", "trained", "compare"],
                        default="compare", help="Demo mode")
    parser.add_argument("--tickets", type=int, default=TICKETS_PER_TASK,
                        help="Tickets per task")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    p = lambda msg: print(msg, flush=True)

    p("=" * 60)
    p("  FLEET AI - SCALABLE OVERSIGHT DEMO")
    p("=" * 60)
    p(f"  Mode: {args.mode}")
    p(f"  Model: {MODEL_NAME}")
    p(f"  API Key: {'set' if (API_KEY or OPENAI_API_KEY) else 'NOT SET'}")
    p(f"  Env URL: {ENV_URL}")
    p(f"  Tickets per task: {args.tickets}")
    p("=" * 60)

    p("\n  Checking environment health...")
    try:
        health = http_get(f"{ENV_URL}/health")
        p(f"  Environment: healthy")
    except Exception as e:
        p(f"  WARNING: Could not reach environment: {e}")
        p("  Start the server first: python -m uvicorn server.app:app --port 7860")
        return

    if args.mode in ("baseline", "compare"):
        p(f"\n{'=' * 60}")
        p("  BASELINE (Heuristic Inspector)")
        p(f"{'=' * 60}")
        baseline_results = {}
        for task in TASKS:
            baseline_results[task] = run_task(task, args.tickets, "baseline")
        print_summary(baseline_results, "baseline")
    else:
        baseline_results = {}

    if args.mode in ("trained", "compare"):
        p(f"\n{'=' * 60}")
        p(f"  TRAINED ({MODEL_NAME})")
        p(f"{'=' * 60}")
        if not (API_KEY or OPENAI_API_KEY):
            p("  WARNING: No API key set. Using heuristic fallback.")
            p("  Set API_KEY or OPENAI_API_KEY for LLM-based inspection.")
        trained_results = {}
        for task in TASKS:
            trained_results[task] = run_task(task, args.tickets, "trained")
        print_summary(trained_results, "trained")
    else:
        trained_results = {}

    if args.mode == "compare":
        print_comparison(baseline_results, trained_results)

    output = {
        "mode": args.mode,
        "model": MODEL_NAME,
        "seed": args.seed,
        "tickets_per_task": args.tickets,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": {
            task: {"mean_score": round(r["mean_score"], 4), "scores": r["scores"]}
            for task, r in baseline_results.items()
        } if baseline_results else None,
        "trained": {
            task: {"mean_score": round(r["mean_score"], 4), "scores": r["scores"]}
            for task, r in trained_results.items()
        } if trained_results else None,
    }

    output_path = "demo_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    p(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
