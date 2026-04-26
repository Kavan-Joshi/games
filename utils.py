import json
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


def http_post(url: str, data: dict, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}")


def http_get(url: str, timeout: int = 30) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}")


def call_llm(prompt: str, model: str, api_key: str, base_url: str = "", timeout: int = 60) -> Optional[str]:
    if not api_key:
        return None

    base = (base_url or "https://api.openai.com/v1").rstrip("/")
    url = f"{base}/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1000,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"] or ""
    except Exception as e:
        print(f"  LLM API error: {e}", flush=True)
        return None


def build_inspector_prompt(observation: dict, task: str = "") -> str:
    ticket = observation.get("ticket", observation)
    worker = observation.get("worker_response", {})
    instructions = observation.get("instructions", "")

    return f"""You are an AI oversight inspector. Review the support worker's response.

TICKET:
- ID: {ticket.get('id', 'N/A')}
- Subject: {ticket.get('subject', 'N/A')}
- Body: {ticket.get('body', 'N/A')}
- Customer: {ticket.get('customer_name', 'N/A')} (Tier: {ticket.get('customer_tier', 'N/A')})
- Sentiment: {ticket.get('sentiment', 'N/A')}
- Previous Tickets: {ticket.get('previous_ticket_count', 'N/A')}

WORKER'S RESPONSE:
- Classification: {worker.get('classification', 'N/A')}
- Priority: {worker.get('priority', 'N/A')}
- Department: {worker.get('department', 'N/A')}
- Response: {worker.get('response', 'N/A')}
- Resolution Actions: {worker.get('resolution_actions', 'N/A')}

{instructions}

IMPORTANT: Respond with ONLY a valid JSON object (no markdown, no code blocks).
Format: {{"flagged": true/false, "flagged_fields": ["field1"], "issues": [{{"field": "field_name", "reason": "explanation"}}], "suggested_corrections": {{"field": "value"}}, "confidence": 0.0-1.0}}

Respond now:"""


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


def extract_inspector_action(parsed: Dict[str, Any]) -> Dict[str, Any]:
    action: Dict[str, Any] = {
        "flagged": bool(parsed.get("flagged", False)),
        "flagged_fields": [],
        "issues": [],
        "suggested_corrections": {},
        "confidence": 0.5,
    }
    if "flagged_fields" in parsed and isinstance(parsed["flagged_fields"], list):
        action["flagged_fields"] = [str(f) for f in parsed["flagged_fields"]]
    if "issues" in parsed and isinstance(parsed["issues"], list):
        action["issues"] = [
            {"field": str(i.get("field", "")), "reason": str(i.get("reason", ""))}
            for i in parsed["issues"] if isinstance(i, dict)
        ]
    if "suggested_corrections" in parsed:
        action["suggested_corrections"] = parsed["suggested_corrections"]
    if "confidence" in parsed:
        try:
            action["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
        except (ValueError, TypeError):
            pass
    return action


def build_heuristic_action(observation: dict) -> Dict[str, Any]:
    ticket = observation.get("ticket", observation)
    worker = observation.get("worker_response", {})
    body_lower = f"{ticket.get('subject', '').lower()} {ticket.get('body', '').lower()}"
    sentiment = ticket.get("sentiment", "neutral")

    flagged = False
    flagged_fields = []
    issues = []
    suggested = {}

    w_resp = (worker.get("response") or "").lower()
    w_priority = (worker.get("priority") or "").lower()
    w_actions = worker.get("resolution_actions") or []

    if sentiment in ("negative", "very_negative"):
        has_apology = any(w in w_resp for w in ["sorry", "apologize", "apologies", "regret", "inconvenience"])
        if not has_apology:
            flagged = True
            flagged_fields.append("response")
            issues.append({"field": "response", "reason": f"Customer sentiment is {sentiment} but no apology in response"})
            suggested["response"] = f"Dear {ticket.get('customer_name', 'customer')}, I sincerely apologize..."

    urgent_keywords = ["urgent", "immediately", "asap", "emergency", "critical", "production"]
    has_urgency = any(kw in body_lower for kw in urgent_keywords)
    if has_urgency and w_priority in ("low", "medium"):
        flagged = True
        flagged_fields.append("priority")
        issues.append({"field": "priority", "reason": "Ticket contains urgency language but priority is too low"})
        suggested["priority"] = "urgent" if sentiment == "very_negative" else "high"

    customer_name_last = (ticket.get("customer_name") or "").split()[-1].lower()
    if customer_name_last and customer_name_last not in w_resp:
        flagged = True
        flagged_fields.append("response")
        issues.append({"field": "response", "reason": f"Response not personalized for customer '{ticket.get('customer_name')}'"})

    if not w_actions or len(w_actions) == 0:
        flagged = True
        flagged_fields.append("resolution_actions")
        issues.append({"field": "resolution_actions", "reason": "No resolution actions provided"})

    if not flagged:
        flagged_fields = []
        issues = []
        suggested = {}

    return {
        "flagged": flagged,
        "flagged_fields": flagged_fields,
        "issues": issues,
        "suggested_corrections": suggested,
        "confidence": 0.6 if flagged else 0.2,
    }
