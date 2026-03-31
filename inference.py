import os
import sys
import json
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from environment import CustomerSupportEnv
from environment.models import Action, ResetRequest

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

TICKETS_PER_TASK = 5
SEED = 42

TASK_CONFIGS = {
    "classification": {
        "required_fields": ["classification", "priority"],
        "prompt_fields": "classification (one of: billing, technical, account, product, shipping) and priority (one of: low, medium, high, urgent)",
    },
    "routing_response": {
        "required_fields": ["classification", "priority", "department", "response"],
        "prompt_fields": "classification, priority, department (one of: finance, engineering, account_management, product_team, logistics, billing_support, technical_support, general_support), and a professional response string",
    },
    "full_resolution": {
        "required_fields": ["classification", "priority", "department", "response", "resolution_actions"],
        "prompt_fields": "classification, priority, department, a professional response string (that references customer history when available), and resolution_actions (a list of specific action strings)",
    },
}


def build_prompt(observation: dict, task: str) -> str:
    ticket = observation["ticket"]
    instructions = observation["instructions"]

    prompt = f"""You are a professional customer support agent. Your task is to handle the following support ticket.

TICKET INFORMATION:
- Ticket ID: {ticket['id']}
- Subject: {ticket['subject']}
- Body: {ticket['body']}
- Customer: {ticket['customer_name']}
- Customer Tier: {ticket['customer_tier']}
- Customer Tenure: {ticket['customer_tenure_days']} days
- Sentiment: {ticket['sentiment']}
- Previous Tickets: {ticket['previous_ticket_count']}"""

    if observation.get("customer_history"):
        ch = observation["customer_history"]
        prompt += f"""

CUSTOMER HISTORY:
- Total Tickets: {ch['total_tickets']}
- Resolved Tickets: {ch['resolved_tickets']}
- Avg Satisfaction: {ch['avg_satisfaction_score']}/5.0
- Lifetime Value: ${ch['lifetime_value_usd']:,.2f}
- Recent Issues: {', '.join(ch['recent_issues']) if ch['recent_issues'] else 'None'}
- Last Contact: {ch['last_contact_days_ago']} days ago
- Escalation History: {', '.join(ch['escalation_history']) if ch['escalation_history'] else 'None'}"""

    config = TASK_CONFIGS[task]
    prompt += f"""

INSTRUCTIONS:
{instructions}

IMPORTANT: You MUST respond with ONLY a valid JSON object (no markdown, no code blocks, no explanation).
The JSON must contain: {config['prompt_fields']}.

Example response format:
{{"classification": "billing", "priority": "high", "department": "billing_support", "response": "Dear customer, ...", "resolution_actions": ["action1", "action2"]}}

Respond with the JSON object now:"""

    return prompt


def parse_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def extract_action(parsed: Dict[str, Any], task: str) -> Action:
    config = TASK_CONFIGS[task]
    action = Action()
    if "classification" in parsed:
        action.classification = str(parsed["classification"]).lower().strip()
    if "priority" in parsed:
        action.priority = str(parsed["priority"]).lower().strip()
    if "department" in parsed:
        action.department = str(parsed["department"]).lower().strip().replace(" ", "_")
    if "response" in parsed:
        action.response = str(parsed["response"])
    if "resolution_actions" in parsed:
        actions = parsed["resolution_actions"]
        if isinstance(actions, list):
            action.resolution_actions = [str(a) for a in actions if a]
        elif isinstance(actions, str):
            action.resolution_actions = [actions]

    for field_name in config["required_fields"]:
        val = getattr(action, field_name)
        if val is None:
            pass

    return action


def run_task(env: CustomerSupportEnv, task: str, client: OpenAI, num_tickets: int) -> Dict[str, Any]:
    scores = []
    details = []

    for i in range(num_tickets):
        reset_request = ResetRequest(task=task, seed=SEED + i)
        reset_result = env.reset(reset_request)
        observation = reset_result.observation
        ticket_id = reset_result.info.get("ticket_id", "unknown")

        prompt = build_prompt(observation.model_dump(), task)

        try:
            api_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
            )
            response_text = api_response.choices[0].message.content or ""
        except Exception as e:
            details.append({
                "ticket_id": ticket_id,
                "error": str(e),
                "score": 0.0,
            })
            scores.append(0.0)
            continue

        parsed = parse_json_from_response(response_text)

        if not parsed:
            action = Action(response=response_text[:500])
            details.append({
                "ticket_id": ticket_id,
                "parse_error": True,
                "raw_response": response_text[:200],
                "score": 0.0,
            })
            scores.append(0.0)
        else:
            action = extract_action(parsed, task)
            step_result = env.step(action)
            score = step_result.reward
            grader_info = step_result.info.get("grader", {})

            detail = {
                "ticket_id": ticket_id,
                "score": score,
                "parsed_action": {
                    "classification": action.classification,
                    "priority": action.priority,
                    "department": action.department,
                    "response_length": len(action.response) if action.response else 0,
                    "resolution_actions_count": len(action.resolution_actions) if action.resolution_actions else 0,
                },
                "grader_components": grader_info.get("components", []),
            }
            details.append(detail)
            scores.append(score)

    return {
        "task": task,
        "num_tickets": num_tickets,
        "scores": scores,
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "details": details,
    }


def main():
    print("=" * 60)
    print("  CUSTOMER SUPPORT TRIAGE - BASELINE INFERENCE")
    print("=" * 60)
    print(f"  Model: {MODEL_NAME}")
    print(f"  API Key: {'set' if OPENAI_API_KEY else 'NOT SET'}")
    print(f"  Tickets per task: {TICKETS_PER_TASK}")
    print(f"  Seed: {SEED}")
    print("=" * 60)

    if not OPENAI_API_KEY:
        print("\nERROR: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("Also set MODEL_NAME (default: gpt-4o-mini)")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)
    env = CustomerSupportEnv()

    all_results = {}
    start_time = time.time()

    for task in ["classification", "routing_response", "full_resolution"]:
        print(f"\n{'='*60}")
        print(f"  Running task: {task}")
        print(f"{'='*60}")

        result = run_task(env, task, client, TICKETS_PER_TASK)
        all_results[task] = result

        print(f"\n  Scores: {[f'{s:.3f}' for s in result['scores']]}")
        print(f"  Mean:   {result['mean_score']:.3f}")
        print(f"  Range:  [{result['min_score']:.3f}, {result['max_score']:.3f}]")

        for d in result["details"]:
            status = "OK" if d.get("score", 0) >= 0.5 else "LOW"
            if d.get("error"):
                status = "ERROR"
            elif d.get("parse_error"):
                status = "PARSE_ERR"
            print(f"    {d.get('ticket_id', '?')}: {d.get('score', 0):.3f} [{status}]")

    elapsed = time.time() - start_time

    print(f"\n{'#'*60}")
    print(f"  SUMMARY")
    print(f"{'#'*60}")

    overall_scores = []
    for task, result in all_results.items():
        print(f"  {task:25s}: mean={result['mean_score']:.3f}  range=[{result['min_score']:.3f}, {result['max_score']:.3f}]")
        overall_scores.extend(result["scores"])

    overall_mean = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    print(f"\n  OVERALL MEAN SCORE: {overall_mean:.3f}")
    print(f"  TOTAL RUNTIME: {elapsed:.1f}s")
    print(f"{'#'*60}")

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

    output_path = os.environ.get("OUTPUT_PATH", "baseline_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
