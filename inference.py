import os
import sys
import json
import re
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

ENV_URL = os.environ.get("ENV_URL", "https://kavanjoshi-openenv.hf.space")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")

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


def http_post(url: str, data: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}")


def http_get(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}")


def call_openai(prompt: str) -> Optional[str]:
    try:
        from openai import OpenAI

        api_key = API_KEY or OPENAI_API_KEY
        base_url = API_BASE_URL or None

        if not api_key:
            return None

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
        )
        return response.choices[0].message.content or ""
    except ImportError:
        return None
    except Exception as e:
        print(f"  OpenAI API error: {e}", flush=True)
        return None


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


def build_action_from_parsed(parsed: Dict[str, Any], task: str) -> Dict[str, Any]:
    action: Dict[str, Any] = {}
    if "classification" in parsed:
        action["classification"] = str(parsed["classification"]).lower().strip()
    if "priority" in parsed:
        action["priority"] = str(parsed["priority"]).lower().strip()
    if "department" in parsed:
        action["department"] = str(parsed["department"]).lower().strip().replace(" ", "_")
    if "response" in parsed:
        action["response"] = str(parsed["response"])
    if "resolution_actions" in parsed:
        actions = parsed["resolution_actions"]
        if isinstance(actions, list):
            action["resolution_actions"] = [str(a) for a in actions if a]
        elif isinstance(actions, str):
            action["resolution_actions"] = [actions]
    return action


def build_fallback_action(observation: dict, task: str) -> Dict[str, Any]:
    ticket = observation["ticket"]
    action: Dict[str, Any] = {}

    body_lower = ticket["body"].lower()
    subject_lower = ticket["subject"].lower()

    billing_words = ["charge", "payment", "bill", "refund", "invoice", "credit card", "subscription", "billing", "price", "cost", "fee"]
    technical_words = ["bug", "error", "crash", "not working", "broken", "glitch", "api", "integration", "install", "update"]
    account_words = ["password", "login", "account", "access", "locked", "signup", "delete account", "security"]
    product_words = ["feature", "plan", "upgrade", "comparison", "how to", "does it"]
    shipping_words = ["shipping", "delivery", "order", "package", "tracking", "return", "arrived"]

    text = f"{subject_lower} {body_lower}"
    scores = {
        "billing": sum(1 for w in billing_words if w in text),
        "technical": sum(1 for w in technical_words if w in text),
        "account": sum(1 for w in account_words if w in text),
        "product": sum(1 for w in product_words if w in text),
        "shipping": sum(1 for w in shipping_words if w in text),
    }
    action["classification"] = max(scores, key=scores.get) if max(scores.values()) > 0 else "general"

    if any(w in text for w in ["urgent", "immediately", "asap", "cannot", "unable", "down", "lost"]):
        action["priority"] = "high"
    elif any(w in text for w in ["frustrated", "angry", "terrible", "unacceptable"]):
        action["priority"] = "urgent"
    else:
        action["priority"] = "medium"

    dept_map = {
        "billing": "billing_support",
        "technical": "technical_support",
        "account": "account_management",
        "product": "product_team",
        "shipping": "logistics",
    }
    action["department"] = dept_map.get(action["classification"], "general_support")

    name = ticket.get("customer_name", "customer")
    action["response"] = (
        f"Dear {name},\n\nThank you for contacting our support team. "
        f"We have received your ticket regarding '{ticket['subject']}'. "
        f"We are looking into this matter and will get back to you shortly. "
        f"We appreciate your patience.\n\nBest regards,\nCustomer Support Team"
    )

    if task == "full_resolution":
        action["resolution_actions"] = [
            "review_ticket_details",
            "investigate_root_cause",
            "provide_resolution",
            "follow_up_with_customer",
        ]

    return action


def run_task(task: str, num_tickets: int) -> Dict[str, Any]:
    scores = []
    details = []
    has_openai = False

    try:
        from openai import OpenAI
        has_openai = True
    except ImportError:
        pass

    has_api_key = bool(API_KEY or OPENAI_API_KEY)

    print(f"[START] task={task}", flush=True)

    for i in range(num_tickets):
        ticket_id = "unknown"
        try:
            reset_result = http_post(f"{ENV_URL}/reset", {"task": task, "seed": SEED + i})
            observation = reset_result.get("observation", reset_result)
            ticket_id = reset_result.get("info", {}).get("ticket_id", "unknown")
        except Exception as e:
            print(f"    Reset error: {e}", flush=True)
            scores.append(0.0)
            details.append({"ticket_id": ticket_id, "error": str(e), "score": 0.0})
            print(f"[STEP] step={i+1} reward=0.0", flush=True)
            continue

        prompt = build_prompt(observation, task)
        action = None

        if has_openai and has_api_key:
            response_text = call_openai(prompt)
            if response_text:
                parsed = parse_json_from_response(response_text)
                if parsed:
                    action = build_action_from_parsed(parsed, task)

        if action is None:
            action = build_fallback_action(observation, task)

        score = 0.0
        try:
            step_result = http_post(f"{ENV_URL}/step", action)
            score = step_result.get("reward", 0.0)
            grader_info = step_result.get("info", {}).get("grader", {})

            detail = {
                "ticket_id": ticket_id,
                "score": score,
                "parsed_action": {
                    "classification": action.get("classification"),
                    "priority": action.get("priority"),
                    "department": action.get("department"),
                    "response_length": len(action.get("response", "")),
                    "resolution_actions_count": len(action.get("resolution_actions", [])),
                },
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
    p("  CUSTOMER SUPPORT TRIAGE - BASELINE INFERENCE")
    p("=" * 60)
    p(f"  Model: {MODEL_NAME}")
    p(f"  API Key: {'set' if OPENAI_API_KEY else 'NOT SET'}")
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

    for task in ["classification", "routing_response", "full_resolution"]:
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
            p(f"    {d.get('ticket_id', '?')}: {d.get('score', 0):.3f} [{status}]")

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

    output_path = os.environ.get("OUTPUT_PATH", "baseline_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    p(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
