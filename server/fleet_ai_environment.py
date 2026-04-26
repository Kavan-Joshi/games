import os
import random
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State
from fastmcp import FastMCP

from environment.error_injector import ErrorInjector
from environment.env import (
    ERROR_MIX_BY_TASK, TASK_INSTRUCTIONS, MIN_REWARD_FLOOR,
    MAX_STEPS, EPISODE_TIMEOUT_SECONDS, _generate_hints, _shape_reward,
)
from environment.graders import grade_inspector
from environment.models import (
    ErrorType, InjectedError, InspectorAction, TaskType,
)
from environment.tickets import TICKETS, TicketRecord

_episodes: Dict[str, Dict[str, Any]] = {}


class FleetAIEnvironment(MCPEnvironment):
    """
    FleetAI - Scalable Oversight Environment.

    An MCP environment for training AI inspector agents that monitor,
    audit, and correct customer support worker agents.

    Supports multi-step episodes with progressive hints and retry.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        mcp = FastMCP("fleet_ai")

        @mcp.tool
        def inspect_ticket(
            episode_id: str,
            flagged: bool,
            flagged_fields: List[str],
            issues: List[Dict[str, str]],
            suggested_corrections: Dict[str, Any],
            confidence: float,
        ) -> dict:
            """
            Submit an inspection review for a worker's response.
            Up to 3 attempts allowed per episode. Hints provided between attempts.

            Args:
                episode_id: The episode ID from the reset response
                flagged: Whether the worker response contains errors
                flagged_fields: List of fields with errors
                issues: List of dicts with 'field' and 'reason' keys
                suggested_corrections: Dict mapping field names to corrected values
                confidence: Confidence level (0.0-1.0)

            Returns:
                Grading result with reward, component scores, and hints for next attempt
            """
            return self._handle_inspect(episode_id, flagged, flagged_fields, issues, suggested_corrections, confidence)

        @mcp.tool
        def get_ticket_info(episode_id: str) -> dict:
            """
            Get details about the current ticket and worker response.

            Args:
                episode_id: The episode ID from the reset response

            Returns:
                Ticket and worker response details
            """
            return self._handle_get_ticket_info(episode_id)

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: str = "inspection_easy",
        **kwargs: Any,
    ) -> Observation:
        valid_tasks = [t.value for t in TaskType]
        if task not in valid_tasks:
            task = TaskType.INSPECTION_EASY.value

        rng = random.Random(seed) if seed is not None else random.Random()
        error_injector = ErrorInjector(rng=rng)

        eligible = [t for t in TICKETS if t.ground_truth.difficulty <= 3]
        ticket = rng.choice(eligible)

        error_mix = ERROR_MIX_BY_TASK.get(task, ERROR_MIX_BY_TASK[TaskType.INSPECTION_EASY.value])
        worker_response, injected_errors = error_injector.generate_worker_response(
            ticket, error_mix=error_mix
        )

        eid = episode_id or str(uuid4())

        _episodes[eid] = {
            "ticket": ticket,
            "worker_response": worker_response,
            "injected_errors": injected_errors,
            "task": task,
            "done": False,
            "step_count": 0,
            "best_reward": 0.0,
            "step_rewards": [],
            "cumulative_hints": [],
            "last_grader_result": None,
            "start_time": time.time(),
        }

        self._state = State(episode_id=eid, step_count=0)

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "episode_id": eid,
                "task_id": ticket.id,
                "task_type": task,
                "ticket": {
                    "id": ticket.id,
                    "subject": ticket.subject,
                    "body": ticket.body,
                    "customer_id": ticket.customer_id,
                    "customer_name": ticket.customer_name,
                    "customer_tier": ticket.customer_tier,
                    "customer_tenure_days": ticket.customer_tenure_days,
                    "sentiment": ticket.sentiment,
                    "previous_ticket_count": ticket.previous_ticket_count,
                },
                "worker_response": worker_response.model_dump(),
                "has_injected_errors": len(injected_errors) > 0,
                "num_injected_errors": len(injected_errors),
                "injected_error_fields": [e.field for e in injected_errors],
                "instructions": TASK_INSTRUCTIONS.get(task, ""),
                "max_steps": MAX_STEPS,
                "hints": [],
            },
        )

    def _handle_get_ticket_info(self, episode_id: str) -> dict:
        episode = _episodes.get(episode_id)
        if not episode:
            return {"error": f"Episode {episode_id} not found. Call reset first."}

        ticket = episode["ticket"]
        worker = episode["worker_response"]
        return {
            "episode_id": episode_id,
            "task": episode["task"],
            "ticket": {
                "id": ticket.id,
                "subject": ticket.subject,
                "body": ticket.body,
                "customer_name": ticket.customer_name,
                "customer_tier": ticket.customer_tier,
                "sentiment": ticket.sentiment,
            },
            "worker_response": worker.model_dump(),
            "step_count": episode["step_count"],
            "best_reward": episode["best_reward"],
            "cumulative_hints": episode["cumulative_hints"],
            "max_steps": MAX_STEPS,
        }

    def _handle_inspect(
        self,
        episode_id: str,
        flagged: bool,
        flagged_fields: List[str],
        issues: List[Dict[str, str]],
        suggested_corrections: Dict[str, Any],
        confidence: float,
    ) -> dict:
        episode = _episodes.get(episode_id)
        if not episode:
            return {"error": f"Episode {episode_id} not found. Call reset first."}

        if episode["done"]:
            return {
                "error": "Episode already completed. Start a new episode with reset.",
                "best_reward": episode["best_reward"],
            }

        episode["step_count"] += 1
        step = episode["step_count"]

        elapsed = time.time() - episode["start_time"]
        if elapsed > EPISODE_TIMEOUT_SECONDS:
            episode["done"] = True
            shaped = _shape_reward(episode["best_reward"], step, MAX_STEPS)
            return {
                "reward": shaped,
                "done": True,
                "error": f"Episode timed out after {elapsed:.1f}s",
                "penalty": "timeout",
                "best_reward": episode["best_reward"],
            }

        ticket = episode["ticket"]
        injected_errors = episode["injected_errors"]
        task = episode["task"]

        action = InspectorAction(
            flagged=flagged,
            flagged_fields=flagged_fields,
            issues=issues,
            suggested_corrections=suggested_corrections,
            confidence=max(0.0, min(1.0, confidence)),
        )

        grader_result = grade_inspector(
            injected_errors=injected_errors,
            inspector=action,
            ticket=ticket,
            task_type=task,
        )

        raw_reward = grader_result.overall_score
        episode["step_rewards"].append(raw_reward)
        episode["last_grader_result"] = grader_result

        if raw_reward > episode["best_reward"]:
            episode["best_reward"] = raw_reward

        is_final = step >= MAX_STEPS
        is_excellent = raw_reward >= 0.8

        if is_final or is_excellent:
            episode["done"] = True
            shaped = _shape_reward(episode["best_reward"], step, MAX_STEPS)

            result = {
                "reward": shaped,
                "done": True,
                "raw_reward": raw_reward,
                "best_reward": episode["best_reward"],
                "task": task,
                "ticket_id": ticket.id,
                "step": step,
                "total_steps": step,
                "score_progression": [round(s, 4) for s in episode["step_rewards"]],
                "ground_truth": {
                    "category": ticket.ground_truth.category,
                    "priority": ticket.ground_truth.priority,
                    "department": ticket.ground_truth.department,
                },
                "injected_errors": [
                    {
                        "field": e.field,
                        "type": e.error_type.value,
                        "ground_truth": e.ground_truth_value,
                        "corrupted": e.corrupted_value,
                        "subtlety": e.subtlety,
                    }
                    for e in injected_errors
                ],
                "grader": {
                    "overall_score": grader_result.overall_score,
                    "components": [
                        {"name": cs.name, "score": cs.score, "max_weight": cs.max_weight, "details": cs.details}
                        for cs in grader_result.component_scores
                    ],
                },
                "hints_received": episode["cumulative_hints"],
            }
            return result

        new_hints = _generate_hints(step, injected_errors, grader_result, ticket)
        episode["cumulative_hints"].extend(new_hints)

        shaped_interim = _shape_reward(raw_reward, step, MAX_STEPS)

        return {
            "reward": shaped_interim,
            "done": False,
            "can_retry": True,
            "remaining_attempts": MAX_STEPS - step,
            "current_score": round(raw_reward, 4),
            "best_score_so_far": round(episode["best_reward"], 4),
            "task": task,
            "ticket_id": ticket.id,
            "step": step,
            "hints": new_hints,
            "grader": {
                "overall_score": grader_result.overall_score,
                "components": [
                    {"name": cs.name, "score": cs.score, "max_weight": cs.max_weight, "details": cs.details}
                    for cs in grader_result.component_scores
                ],
            },
        }

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions.",
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._state
