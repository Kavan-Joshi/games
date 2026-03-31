import re
import string
from typing import Dict, List, Optional, Tuple

from .models import GraderComponentScore, GraderResult
from .tickets import TicketRecord, TicketGroundTruth, CustomerHistoryRecord


PRIORITY_LEVELS = {"low": 0, "medium": 1, "high": 2, "urgent": 3}

DEPARTMENT_ALIASES = {
    "billing": ["billing", "billing_support", "finance"],
    "technical": ["technical", "technical_support", "engineering"],
    "account": ["account", "account_management"],
    "product": ["product", "product_team", "general_support"],
    "shipping": ["shipping", "logistics", "delivery"],
}

CATEGORY_ALIASES = {
    "billing": ["billing", "payment", "invoice", "refund", "charge"],
    "technical": ["technical", "bug", "crash", "api", "integration", "webhook", "sso"],
    "account": ["account", "password", "login", "access", "delete", "lock", "security"],
    "product": ["product", "feature", "plan", "integration_inquiry", "comparison", "api_inquiry"],
    "shipping": ["shipping", "delivery", "order", "return", "package", "customs"],
}

TOXIC_PATTERNS = [
    r"\bfuck\b", r"\bshit\b", r"\bdamn\b", r"\bast\b", r"\bidiot\b",
    r"\bstupid\b", r"\bdumb\b", r"\bhate\b", r"\bkys\b", r"\bkill\b",
    r"\bthreat\b", r"\bsue\b.*\byou\b",
]


def _normalize(text: str) -> str:
    return text.lower().strip().rstrip(".")


def _score_classification(agent_value: Optional[str], ground_truth: str) -> GraderComponentScore:
    if not agent_value:
        return GraderComponentScore(name="classification", score=0.0, max_weight=1.0, details="No classification provided")

    normalized = _normalize(agent_value)
    gt = _normalize(ground_truth)

    if normalized == gt:
        return GraderComponentScore(name="classification", score=1.0, max_weight=1.0, details="Exact match")

    gt_aliases = CATEGORY_ALIASES.get(gt, [])
    if normalized in gt_aliases:
        return GraderComponentScore(name="classification", score=0.5, max_weight=1.0, details=f"Close match: '{agent_value}' maps to '{ground_truth}'")

    for alias in gt_aliases:
        if alias in normalized or normalized in alias:
            return GraderComponentScore(name="classification", score=0.3, max_weight=1.0, details=f"Partial match: '{agent_value}' related to '{ground_truth}'")

    return GraderComponentScore(name="classification", score=0.0, max_weight=1.0, details=f"Wrong: '{agent_value}' vs '{ground_truth}'")


def _score_priority(agent_value: Optional[str], ground_truth: str) -> GraderComponentScore:
    if not agent_value:
        return GraderComponentScore(name="priority", score=0.0, max_weight=1.0, details="No priority provided")

    normalized = _normalize(agent_value)
    gt = _normalize(ground_truth)

    if normalized == gt:
        return GraderComponentScore(name="priority", score=1.0, max_weight=1.0, details="Exact match")

    agent_level = PRIORITY_LEVELS.get(normalized, -1)
    gt_level = PRIORITY_LEVELS.get(gt, -1)

    if agent_level >= 0 and gt_level >= 0:
        diff = abs(agent_level - gt_level)
        if diff == 1:
            return GraderComponentScore(name="priority", score=0.5, max_weight=1.0, details=f"One level off: '{agent_value}' vs '{ground_truth}'")
        if diff == 2:
            return GraderComponentScore(name="priority", score=0.2, max_weight=1.0, details=f"Two levels off: '{agent_value}' vs '{ground_truth}'")

    return GraderComponentScore(name="priority", score=0.0, max_weight=1.0, details=f"Wrong: '{agent_value}' vs '{ground_truth}'")


def _score_department(agent_value: Optional[str], ground_truth: str) -> GraderComponentScore:
    if not agent_value:
        return GraderComponentScore(name="department", score=0.0, max_weight=1.0, details="No department provided")

    normalized = _normalize(agent_value)
    gt = _normalize(ground_truth)

    if normalized == gt:
        return GraderComponentScore(name="department", score=1.0, max_weight=1.0, details="Exact match")

    gt_aliases = DEPARTMENT_ALIASES.get(gt, [])
    if normalized in gt_aliases:
        return GraderComponentScore(name="department", score=0.7, max_weight=1.0, details=f"Alias match: '{agent_value}' maps to '{ground_truth}'")

    for cat, aliases in DEPARTMENT_ALIASES.items():
        if normalized in aliases:
            if cat == gt or gt in aliases:
                return GraderComponentScore(name="department", score=0.5, max_weight=1.0, details=f"Related department: '{agent_value}' vs '{ground_truth}'")

    return GraderComponentScore(name="department", score=0.0, max_weight=1.0, details=f"Wrong: '{agent_value}' vs '{ground_truth}'")


def _check_toxicity(text: str) -> Tuple[bool, float]:
    text_lower = text.lower()
    matches = 0
    for pattern in TOXIC_PATTERNS:
        if re.search(pattern, text_lower):
            matches += 1
    is_toxic = matches > 0
    penalty = min(0.3, matches * 0.15)
    return is_toxic, penalty


def _score_response(
    agent_response: Optional[str],
    ticket: TicketRecord,
    customer_history: Optional[CustomerHistoryRecord],
    check_context: bool = False,
) -> GraderComponentScore:
    if not agent_response:
        return GraderComponentScore(name="response", score=0.0, max_weight=1.0, details="No response provided")

    is_toxic, toxicity_penalty = _check_toxicity(agent_response)
    if is_toxic:
        return GraderComponentScore(name="response", score=0.0, max_weight=1.0, details="Response contains inappropriate content")

    response_lower = agent_response.lower()
    gt = ticket.ground_truth
    scores = {}
    details = []

    term_hits = sum(1 for term in gt.key_terms if term.lower() in response_lower)
    term_coverage = term_hits / max(len(gt.key_terms), 1)
    scores["key_terms"] = min(term_coverage * 1.2, 1.0)
    details.append(f"Key terms: {term_hits}/{len(gt.key_terms)} matched")

    has_greeting = bool(re.search(r"\b(hi|hello|dear|greetings|good\s+(morning|afternoon|evening))\b", response_lower))
    scores["greeting"] = 1.0 if has_greeting else 0.0
    details.append(f"Greeting: {'yes' if has_greeting else 'no'}")

    needs_apology = ticket.sentiment in ("negative", "very_negative")
    has_apology = bool(re.search(r"\b(sorry|apologize|apologies|regret|inconvenience|understand\s+your\s+frustration)\b", response_lower))
    if needs_apology:
        scores["apology"] = 1.0 if has_apology else 0.0
        details.append(f"Apology (needed): {'yes' if has_apology else 'MISSING'}")
    else:
        scores["apology"] = 1.0
        details.append("Apology: not required")

    has_closing = bool(re.search(r"\b(please\s+let\s+me\s+know|feel\s+free|best\s+regards|regards|thank\s+you|sincerely|cheers)\b", response_lower))
    scores["closing"] = 0.5 if has_closing else 0.0
    details.append(f"Closing: {'yes' if has_closing else 'no'}")

    action_words = ["will", "can", "able to", "going to", "here's how", "steps", "next"]
    has_actionable = any(w in response_lower for w in action_words)
    scores["actionable"] = 0.8 if has_actionable else 0.2
    details.append(f"Actionable: {'yes' if has_actionable else 'vague'}")

    word_count = len(agent_response.split())
    if 20 <= word_count <= 500:
        scores["length"] = 1.0
    elif 10 <= word_count < 20:
        scores["length"] = 0.5
    elif word_count > 500:
        scores["length"] = 0.7
    else:
        scores["length"] = 0.1
    details.append(f"Length: {word_count} words")

    has_customer_name = ticket.customer_name.lower().split()[-1] in response_lower if ticket.customer_name else False
    scores["personalization"] = 1.0 if has_customer_name else 0.3
    details.append(f"Personalization: {'yes' if has_customer_name else 'generic'}")

    weights = {
        "key_terms": 0.30,
        "greeting": 0.08,
        "apology": 0.10,
        "closing": 0.05,
        "actionable": 0.20,
        "length": 0.07,
        "personalization": 0.20,
    }

    if check_context and customer_history:
        context_scores = _score_context_usage(response_lower, ticket, customer_history)
        context_avg = sum(context_scores.values()) / max(len(context_scores), 1)
        scores["context"] = context_avg
        for k, v in context_scores.items():
            details.append(f"Context-{k}: {v:.1f}")
        weights["context"] = 0.15
        for k in list(weights.keys()):
            if k != "context":
                weights[k] *= 0.85

    total_weight = sum(weights.values())
    weighted_score = sum(scores.get(k, 0.0) * weights.get(k, 0.0) for k in weights) / total_weight
    weighted_score = max(0.0, min(1.0, weighted_score - toxicity_penalty))

    return GraderComponentScore(
        name="response",
        score=round(weighted_score, 4),
        max_weight=1.0,
        details="; ".join(details),
    )


def _score_context_usage(
    response_lower: str,
    ticket: TicketRecord,
    history: CustomerHistoryRecord,
) -> Dict[str, float]:
    scores = {}

    if history.total_tickets > 3:
        loyalty_terms = ["valued customer", "long-time", "loyal", "thank you for your continued",
                          "appreciate your business", "since", "customer for"]
        has_loyalty = any(t in response_lower for t in loyalty_terms)
        scores["loyalty"] = 1.0 if has_loyalty else 0.2
    else:
        scores["loyalty"] = 0.5

    if history.recent_issues:
        recent_mentioned = any(issue.lower() in response_lower for issue in history.recent_issues)
        scores["recent_issues"] = 1.0 if recent_mentioned else 0.3
    else:
        scores["recent_issues"] = 0.5

    if history.escalation_history:
        escalation_aware = any(e.lower() in response_lower for e in history.escalation_history)
        scores["escalation_awareness"] = 1.0 if escalation_aware else 0.3
    else:
        scores["escalation_awareness"] = 0.5

    if history.lifetime_value_usd > 5000:
        value_aware = any(t in response_lower for t in ["important", "priority", "dedicated", "account manager"])
        scores["value_awareness"] = 1.0 if value_aware else 0.2
    else:
        scores["value_awareness"] = 0.5

    return scores


def _score_resolution(
    agent_actions: Optional[List[str]],
    ticket: TicketRecord,
    customer_history: Optional[CustomerHistoryRecord],
) -> GraderComponentScore:
    if not agent_actions or len(agent_actions) == 0:
        return GraderComponentScore(name="resolution", score=0.0, max_weight=1.0, details="No resolution actions provided")

    gt = ticket.ground_truth
    expected = [a.lower() for a in gt.expected_resolution_actions]
    agent_lower = [a.lower() for a in agent_actions]

    scores = {}
    details = []

    matched = 0
    partial = 0
    for exp in expected:
        exp_words = set(exp.split("_"))
        for ag in agent_lower:
            ag_words = set(ag.replace("_", " ").replace("-", " ").split())
            overlap = exp_words & ag_words
            if overlap and len(overlap) >= len(exp_words) * 0.5:
                partial += 1
                if len(overlap) >= len(exp_words) * 0.8:
                    matched += 1
                break

    if len(expected) > 0:
        scores["action_match"] = matched / len(expected)
        scores["partial_match"] = partial / len(expected) * 0.5
    else:
        scores["action_match"] = 0.5
        scores["partial_match"] = 0.5
    details.append(f"Expected actions matched: {matched}/{len(expected)}")

    if gt.escalation_required:
        escalation_keywords = ["escalat", "urgent", "priority", "senior", "manager", "engineer", "specialist"]
        has_escalation = any(any(k in a.lower() for k in escalation_keywords) for a in agent_actions)
        scores["escalation"] = 1.0 if has_escalation else 0.0
        details.append(f"Escalation (required): {'yes' if has_escalation else 'MISSING'}")
    else:
        scores["escalation"] = 1.0
        details.append("Escalation: not required")

    num_actions = len(agent_actions)
    if 1 <= num_actions <= 5:
        scores["action_count"] = 1.0
    elif num_actions == 0:
        scores["action_count"] = 0.0
    elif num_actions > 5:
        scores["action_count"] = 0.5
    else:
        scores["action_count"] = 0.5
    details.append(f"Action count: {num_actions}")

    if customer_history and customer_history.lifetime_value_usd > 10000:
        premium_actions = ["dedicated", "priority", "expedite", "account manager", "senior"]
        has_premium = any(any(k in a.lower() for k in premium_actions) for a in agent_actions)
        scores["tier_awareness"] = 1.0 if has_premium else 0.3
        details.append(f"Tier awareness (high-value): {'yes' if has_premium else 'MISSING'}")
    else:
        scores["tier_awareness"] = 0.5
        details.append("Tier awareness: standard customer")

    weights = {
        "action_match": 0.40,
        "partial_match": 0.15,
        "escalation": 0.25,
        "action_count": 0.10,
        "tier_awareness": 0.10,
    }

    total_weight = sum(weights.values())
    weighted_score = sum(scores[k] * weights[k] for k in scores) / total_weight
    weighted_score = max(0.0, min(1.0, weighted_score))

    return GraderComponentScore(
        name="resolution",
        score=round(weighted_score, 4),
        max_weight=1.0,
        details="; ".join(details),
    )


def grade_classification(
    ticket: TicketRecord,
    classification: Optional[str],
    priority: Optional[str],
) -> GraderResult:
    scores = []
    scores.append(_score_classification(classification, ticket.ground_truth.category))
    scores.append(_score_priority(priority, ticket.ground_truth.priority))

    weights = [0.6, 0.4]
    total = sum(s.score * w for s, w in zip(scores, weights))
    overall = max(0.0, min(1.0, total))

    return GraderResult(
        component_scores=scores,
        overall_score=round(overall, 4),
        task_type="classification",
    )


def grade_routing_response(
    ticket: TicketRecord,
    classification: Optional[str],
    priority: Optional[str],
    department: Optional[str],
    response: Optional[str],
) -> GraderResult:
    scores = []
    scores.append(_score_classification(classification, ticket.ground_truth.category))
    scores.append(_score_priority(priority, ticket.ground_truth.priority))
    scores.append(_score_department(department, ticket.ground_truth.department))
    scores.append(_score_response(response, ticket, None, check_context=False))

    weights = [0.15, 0.10, 0.20, 0.55]
    total = sum(s.score * w for s, w in zip(scores, weights))
    overall = max(0.0, min(1.0, total))

    return GraderResult(
        component_scores=scores,
        overall_score=round(overall, 4),
        task_type="routing_response",
    )


def grade_full_resolution(
    ticket: TicketRecord,
    classification: Optional[str],
    priority: Optional[str],
    department: Optional[str],
    response: Optional[str],
    resolution_actions: Optional[List[str]],
    customer_history: Optional[CustomerHistoryRecord],
) -> GraderResult:
    scores = []
    scores.append(_score_classification(classification, ticket.ground_truth.category))
    scores.append(_score_priority(priority, ticket.ground_truth.priority))
    scores.append(_score_department(department, ticket.ground_truth.department))
    scores.append(_score_response(response, ticket, customer_history, check_context=True))
    scores.append(_score_resolution(resolution_actions, ticket, customer_history))

    weights = [0.10, 0.05, 0.10, 0.40, 0.35]
    total = sum(s.score * w for s, w in zip(scores, weights))
    overall = max(0.0, min(1.0, total))

    return GraderResult(
        component_scores=scores,
        overall_score=round(overall, 4),
        task_type="full_resolution",
    )
