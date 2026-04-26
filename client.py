from typing import Any, Dict, List, Optional

from openenv.core.mcp_client import MCPToolClient


class FleetAIEnv(MCPToolClient):
    """
    Client for the FleetAI - Scalable Oversight Environment.

    This client provides a simple interface for interacting with the
    FleetAI Environment via MCP tools.

    Tools available:
    - inspect_ticket: Submit an inspection review for a worker response

    Example:
        >>> with FleetAIEnv(base_url="http://localhost:7860") as env:
        ...     result = env.reset(task="inspection_easy", seed=42)
        ...     print(result)
        ...     inspection = env.call_tool(
        ...         "inspect_ticket",
        ...         episode_id=result["episode_id"],
        ...         flagged=True,
        ...         flagged_fields=["priority"],
        ...         issues=[{"field": "priority", "reason": "Should be urgent"}],
        ...         suggested_corrections={"priority": "urgent"},
        ...         confidence=0.85,
        ...     )
        ...     print(f"Reward: {inspection['reward']}")
    """

    def reset(
        self,
        task: str = "inspection_easy",
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Reset the environment and start a new episode.

        Args:
            task: One of 'inspection_easy', 'inspection_hard', 'inspection_adversarial'
            seed: Random seed for reproducibility

        Returns:
            Observation dict with ticket, worker_response, instructions, and episode_id
        """
        return super().reset(task=task, seed=seed, **kwargs)

    def inspect_ticket(
        self,
        episode_id: str,
        flagged: bool,
        flagged_fields: List[str],
        issues: List[Dict[str, str]],
        suggested_corrections: Dict[str, Any],
        confidence: float,
    ) -> Dict[str, Any]:
        """
        Submit an inspection review for a worker's response.

        Args:
            episode_id: The episode ID from the reset response
            flagged: Whether the worker response contains errors
            flagged_fields: List of fields with errors
            issues: List of dicts with 'field' and 'reason' keys
            suggested_corrections: Dict mapping field names to corrected values
            confidence: Confidence level (0.0-1.0)

        Returns:
            Grading result with reward, grader components, and ground truth
        """
        return self.call_tool(
            "inspect_ticket",
            episode_id=episode_id,
            flagged=flagged,
            flagged_fields=flagged_fields,
            issues=issues,
            suggested_corrections=suggested_corrections,
            confidence=confidence,
        )
