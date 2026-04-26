import os
import sys
import json
import time
from typing import Any, Dict, List, Optional

from utils import http_post, http_get, call_llm, build_inspector_prompt, parse_json_from_response, extract_inspector_action, build_heuristic_action

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")

TICKETS_PER_TASK = 5
SEED = 42

TASKS = ["inspection_easy", "inspection_hard", "inspection_adversarial"]


def run_task(task: str, num_tickets: int) -> Dict[str, Any]:
    scores = []
    details = []
    api_key = API_KEY or OPENAI_API_KEY
    print(f"[START] task={task}", flush=True)

    for i in range(num_tickets):
        ticket_id = "unknown"
        try:
            reset_result = http_post(f"{ENV_URL}/reset", {"task": task, "seed": SEED + i})
            observation = reset_result.get("observation", reset_result)
            ticket_id = reset_result.get("info", {}).get("ticket_id", "unknown")
            has_errors = reset_result.get("info", {}).get("has_injected_errors", False)
            error_fields = reset_result.get("info", {}).get("injected_error_fields", [])
        except Exception as e:
            print(f"    Reset error: {e}", flush=True)
            scores.append(0.0)
            details.append({"ticket_id": ticket_id, "error": str(e), "score": 0.0})
            print(f"[STEP] step={i+1} reward=0.0", flush=True)
            continue

        prompt = build_inspector_prompt(observation, task)
        action = None

        response_text = call_llm(prompt, MODEL_NAME, api_key, API_BASE_URL)
        if response_text:
            parsed = parse_json_from_response(response_text)
            if parsed:
                action = extract_inspector_action(parsed)

        if action is None:
            action = build_heuristic_action(observation)

        score = 0.0
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
            details.append({"ticket_id": ticket_id, "error": str(e), "score": 0.0})

        print(f"[STEP] step={i+1} reward={score:.4f}", flush=True)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    print(f"[END] task={task} score={mean_score:.4f} steps={num_tickets}", flush=True)

    return {
        "task": task,
        "num_tickets": num_tickets,
        "scores": scores,
        "mean_score": mean_score,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "details": details,
    }


def main():
    p = lambda msg: print(msg, flush=True)

    p("=" * 60)
    p("  FLEET AI - SCALABLE OVERSIGHT INFERENCE")
    p("=" * 60)
    p(f"  Model: {MODEL_NAME}")
    p(f"  API Key: {'set' if (API_KEY or OPENAI_API_KEY) else 'NOT SET'}")
    p(f"  API Base URL: {API_BASE_URL or 'default (openai)'}")
    p(f"  Env URL: {ENV_URL}")
    p(f"  Tickets per task: {TICKETS_PER_TASK}")
    p(f"  Seed: {SEED}")
    p("=" * 60)

    p("\n  Checking environment health...")
    try:
        health = http_get(f"{ENV_URL}/health")
        p(f"  Environment: {health}")
    except Exception as e:
        p(f"  WARNING: Could not reach environment: {e}")
        p("  Will attempt to continue anyway...")

    all_results = {}
    start_time = time.time()

    for task in TASKS:
        p(f"\n{'='*60}")
        p(f"  Running task: {task}")
        p(f"{'='*60}")

        result = run_task(task, TICKETS_PER_TASK)
        all_results[task] = result

        p(f"\n  Scores: {[f'{s:.3f}' for s in result['scores']]}")
        p(f"  Mean:   {result['mean_score']:.3f}")
        p(f"  Range:  [{result['min_score']:.3f}, {result['max_score']:.3f}]")

        for d in result["details"]:
            status = "OK" if d.get("score", 0) >= 0.5 else "LOW"
            if d.get("error"):
                status = "ERROR"
            p(f"    {d.get('ticket_id', '?')}: {d.get('score', 0):.3f} [{status}] "
              f"(errors={d.get('had_errors')}, flagged={d.get('inspector_flagged')})")

    elapsed = time.time() - start_time

    p(f"\n{'#'*60}")
    p("  SUMMARY")
    p(f"{'#'*60}")

    overall_scores = []
    for task, result in all_results.items():
        p(f"  {task:25s}: mean={result['mean_score']:.3f}  range=[{result['min_score']:.3f}, {result['max_score']:.3f}]")
        overall_scores.extend(result["scores"])

    overall_mean = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    p(f"\n  OVERALL MEAN SCORE: {overall_mean:.3f}")
    p(f"  TOTAL RUNTIME: {elapsed:.1f}s")
    p(f"{'#'*60}")

    output = {
        "model": MODEL_NAME,
        "seed": SEED,
        "tickets_per_task": TICKETS_PER_TASK,
        "elapsed_seconds": round(elapsed, 1),
        "overall_mean_score": round(overall_mean, 4),
        "tasks": {
            task: {
                "mean_score": round(result["mean_score"], 4),
                "scores": result["scores"],
            }
            for task, result in all_results.items()
        },
    }

    output_path = os.environ.get("OUTPUT_PATH", "fleet_ai_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    p(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
