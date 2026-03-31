from .models import (
    Action, CustomerHistory, GraderComponentScore, GraderResult,
    Observation, ResetRequest, ResetResponse, StepResponse, TaskType,
    Ticket,
)
from .tickets import (
    TICKETS, CustomerHistoryRecord, TicketRecord, get_customer_history, get_tickets_by_difficulty,
)
from .graders import grade_classification, grade_full_resolution, grade_routing_response
import random
import time
import hashlib
from typing import Any, Dict, List, Optional


class CustomerSupportEnv:

    def __init__(self):
        self._current_ticket: Optional[TicketRecord] = None
        self._current_history: Optional[CustomerHistoryRecord] = None
        self._current_observation: Optional[Observation] = None
        self._current_task: str = TaskType.CLASSIFICATION.value
        self._step_count: int = 0
        self._done: bool = False
        self._last_result: Optional[StepResponse] = None
        self._rng: random.Random = random.Random(42)
        self._used_ticket_ids: set = set()
        self._task_ticket_queue: Dict[str, List[str]] = {
            TaskType.CLASSIFICATION.value: [],
            TaskType.ROUTING_RESPONSE.value: [],
            TaskType.FULL_RESOLUTION.value: [],
        }

    def reset(self, request: ResetRequest) -> ResetResponse:
        task = request.task
        if task not in [t.value for t in TaskType]:
            task = TaskType.CLASSIFICATION.value

        self._current_task = task
        self._step_count = 0
        self._done = False
        self._last_result = None

        if request.seed is not None:
            self._rng = random.Random(request.seed)

        max_diff = {
            TaskType.CLASSIFICATION.value: 1,
            TaskType.ROUTING_RESPONSE.value: 2,
            TaskType.FULL_RESOLUTION.value: 3,
        }.get(task, 1)

        eligible = get_tickets_by_difficulty(max_diff)
        available = [t for t in eligible if t.id not in self._used_ticket_ids]

        if not available:
            self._used_ticket_ids.clear()
            available = eligible

        ticket = self._rng.choice(available)
        self._used_ticket_ids.add(ticket.id)
        self._current_ticket = ticket
        self._current_history = get_customer_history(ticket.customer_id)

        observation = self._build_observation(ticket)

        self._current_observation = observation

        return ResetResponse(
            observation=observation,
            reward=0.0,
            done=False,
            info={"task": task, "ticket_id": ticket.id},
        )

    def step(self, action: Action) -> StepResponse:
        if self._done:
            if self._last_result:
                return self._last_result
            return StepResponse(
                observation=self._current_observation or self._build_empty_observation(),
                reward=0.0,
                done=True,
                info={"error": "Episode already completed. Call reset() first."},
            )

        self._step_count += 1
        ticket = self._current_ticket
        history = self._current_history

        if not ticket:
            return StepResponse(
                observation=self._build_empty_observation(),
                reward=0.0,
                done=True,
                info={"error": "No active ticket. Call reset() first."},
            )

        if self._step_count > 5:
            self._done = True
            timeout_penalty = 0.0
            return StepResponse(
                observation=self._current_observation,
                reward=timeout_penalty,
                done=True,
                info={"error": "Max steps exceeded", "penalty": "step_limit"},
            )

        if self._current_task == TaskType.CLASSIFICATION.value:
            grader_result = grade_classification(
                ticket=ticket,
                classification=action.classification,
                priority=action.priority,
            )
        elif self._current_task == TaskType.ROUTING_RESPONSE.value:
            grader_result = grade_routing_response(
                ticket=ticket,
                classification=action.classification,
                priority=action.priority,
                department=action.department,
                response=action.response,
            )
        elif self._current_task == TaskType.FULL_RESOLUTION.value:
            grader_result = grade_full_resolution(
                ticket=ticket,
                classification=action.classification,
                priority=action.priority,
                department=action.department,
                response=action.response,
                resolution_actions=action.resolution_actions,
                customer_history=history,
            )
        else:
            grader_result = GraderResult(
                component_scores=[GraderComponentScore(name="error", score=0.0, max_weight=1.0)],
                overall_score=0.0,
                task_type="unknown",
            )

        reward = grader_result.overall_score
        self._done = True

        info = {
            "task": self._current_task,
            "ticket_id": ticket.id,
            "ground_truth": {
                "category": ticket.ground_truth.category,
                "priority": ticket.ground_truth.priority,
                "department": ticket.ground_truth.department,
            },
            "grader": {
                "overall_score": grader_result.overall_score,
                "components": [
                    {"name": cs.name, "score": cs.score, "max_weight": cs.max_weight, "details": cs.details}
                    for cs in grader_result.component_scores
                ],
            },
            "steps": self._step_count,
        }

        feedback_obs = self._build_feedback_observation(ticket, grader_result)

        result = StepResponse(
            observation=feedback_obs,
            reward=round(reward, 4),
            done=True,
            info=info,
        )

        self._last_result = result
        self._current_observation = feedback_obs

        return result

    def state(self) -> Observation:
        if self._current_observation:
            return self._current_observation
        return self._build_empty_observation()

    def _build_observation(self, ticket: TicketRecord) -> Observation:
        instructions = self._get_task_instructions(self._current_task)
        max_steps_map = {
            TaskType.CLASSIFICATION.value: 1,
            TaskType.ROUTING_RESPONSE.value: 1,
            TaskType.FULL_RESOLUTION.value: 1,
        }

        ticket_model = Ticket(
            id=ticket.id,
            subject=ticket.subject,
            body=ticket.body,
            customer_id=ticket.customer_id,
            customer_name=ticket.customer_name,
            customer_tier=ticket.customer_tier,
            customer_tenure_days=ticket.customer_tenure_days,
            sentiment=ticket.sentiment,
            previous_ticket_count=ticket.previous_ticket_count,
            created_at="2024-01-15T10:30:00Z",
        )

        history_model = None
        if self._current_task == TaskType.FULL_RESOLUTION.value and self._current_history:
            h = self._current_history
            history_model = CustomerHistory(
                customer_id=h.customer_id,
                total_tickets=h.total_tickets,
                resolved_tickets=h.resolved_tickets,
                avg_satisfaction_score=h.avg_satisfaction_score,
                recent_issues=h.recent_issues,
                lifetime_value_usd=h.lifetime_value_usd,
                last_contact_days_ago=h.last_contact_days_ago,
                escalation_history=h.escalation_history,
            )

        return Observation(
            task_id=ticket.id,
            task_type=self._current_task,
            ticket=ticket_model,
            instructions=instructions,
            customer_history=history_model,
            step_number=0,
            max_steps=max_steps_map.get(self._current_task, 1),
        )

    def _build_feedback_observation(self, ticket: TicketRecord, grader: GraderResult) -> Observation:
        feedback_parts = ["GRADING RESULTS:"]
        for cs in grader.component_scores:
            status = "PASS" if cs.score >= 0.6 else "NEEDS IMPROVEMENT" if cs.score >= 0.3 else "FAIL"
            feedback_parts.append(f"  - {cs.name}: {cs.score:.2f} ({status}) - {cs.details}")
        feedback_parts.append(f"  OVERALL SCORE: {grader.overall_score:.2f}")
        feedback_parts.append(f"  Ground truth: category={ticket.ground_truth.category}, priority={ticket.ground_truth.priority}, department={ticket.ground_truth.department}")

        instructions = "\n".join(feedback_parts)

        ticket_model = Ticket(
            id=ticket.id,
            subject=ticket.subject,
            body=ticket.body,
            customer_id=ticket.customer_id,
            customer_name=ticket.customer_name,
            customer_tier=ticket.customer_tier,
            customer_tenure_days=ticket.customer_tenure_days,
            sentiment=ticket.sentiment,
            previous_ticket_count=ticket.previous_ticket_count,
            created_at="2024-01-15T10:30:00Z",
        )

        return Observation(
            task_id=ticket.id,
            task_type=self._current_task,
            ticket=ticket_model,
            instructions=instructions,
            step_number=self._step_count,
            max_steps=1,
        )

    def _build_empty_observation(self) -> Observation:
        return Observation(
            task_id="",
            task_type="none",
            ticket=Ticket(
                id="", subject="", body="", customer_id="", customer_name="",
                customer_tier="bronze", customer_tenure_days=0, sentiment="neutral",
                previous_ticket_count=0, created_at="",
            ),
            instructions="No active episode. Call /reset to start.",
            step_number=0,
            max_steps=1,
        )

    def _get_task_instructions(self, task: str) -> str:
        if task == TaskType.CLASSIFICATION.value:
            return (
                "TASK: Classify the support ticket.\n\n"
                "You must provide:\n"
                "1. classification: One of [billing, technical, account, product, shipping]\n"
                "2. priority: One of [low, medium, high, urgent]\n\n"
                "Scoring: classification accuracy (60%) + priority accuracy (40%).\n"
                "Respond with a JSON object: {\"classification\": \"...\", \"priority\": \"...\"}"
            )
        elif task == TaskType.ROUTING_RESPONSE.value:
            return (
                "TASK: Classify, route, and draft a response to the support ticket.\n\n"
                "You must provide:\n"
                "1. classification: One of [billing, technical, account, product, shipping]\n"
                "2. priority: One of [low, medium, high, urgent]\n"
                "3. department: One of [finance, engineering, account_management, product_team, logistics, billing_support, technical_support, general_support]\n"
                "4. response: A professional customer-facing response addressing the issue\n\n"
                "Scoring: classification (15%) + priority (10%) + department (20%) + response quality (55%).\n"
                "Response quality checks: key terms, greeting, apology (if needed), actionable content, appropriate length, personalization.\n"
                "Respond with a JSON object: {\"classification\": \"...\", \"priority\": \"...\", \"department\": \"...\", \"response\": \"...\"}"
            )
        elif task == TaskType.FULL_RESOLUTION.value:
            return (
                "TASK: Full ticket resolution with customer context analysis.\n\n"
                "You must provide:\n"
                "1. classification: One of [billing, technical, account, product, shipping]\n"
                "2. priority: One of [low, medium, high, urgent]\n"
                "3. department: One of [finance, engineering, account_management, product_team, logistics, billing_support, technical_support, general_support]\n"
                "4. response: A professional customer-facing response that references customer history\n"
                "5. resolution_actions: List of specific actions to resolve this ticket (strings)\n\n"
                "Scoring: classification (10%) + priority (5%) + department (10%) + response quality (40%) + resolution plan (35%).\n"
                "Context usage: reference customer loyalty, recent issues, escalation history, and lifetime value in your response.\n"
                "Respond with a JSON object: {\"classification\": \"...\", \"priority\": \"...\", \"department\": \"...\", \"response\": \"...\", \"resolution_actions\": [...]}"
            )
        return "Unknown task type."


env_instance: Optional[CustomerSupportEnv] = None


def get_env() -> CustomerSupportEnv:
    global env_instance
    if env_instance is None:
        env_instance = CustomerSupportEnv()
    return env_instance
