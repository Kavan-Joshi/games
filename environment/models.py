from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
from enum import Enum


class TaskType(str, Enum):
    INSPECTION_EASY = "inspection_easy"
    INSPECTION_HARD = "inspection_hard"
    INSPECTION_ADVERSARIAL = "inspection_adversarial"


class ErrorType(str, Enum):
    CLASSIFICATION_WRONG = "classification_wrong"
    PRIORITY_OFF = "priority_off"
    DEPARTMENT_WRONG = "department_wrong"
    RESPONSE_MISSING_APOLOGY = "response_missing_apology"
    RESPONSE_TOO_VAGUE = "response_too_vague"
    RESPONSE_NO_PERSONALIZATION = "response_no_personalization"
    RESOLUTION_INCOMPLETE = "resolution_incomplete"
    ESCALATION_MISSING = "escalation_missing"
    CLEAN_BUT_SUSPICIOUS = "clean_but_suspicious"


class Ticket(BaseModel):
    id: str
    subject: str
    body: str
    customer_id: str
    customer_name: str
    customer_tier: str
    customer_tenure_days: int
    sentiment: str
    previous_ticket_count: int
    created_at: str


class CustomerHistory(BaseModel):
    customer_id: str
    total_tickets: int
    resolved_tickets: int
    avg_satisfaction_score: float
    recent_issues: List[str]
    lifetime_value_usd: float
    last_contact_days_ago: int
    escalation_history: List[str]


class WorkerResponse(BaseModel):
    classification: Optional[str] = None
    priority: Optional[str] = None
    department: Optional[str] = None
    response: Optional[str] = None
    resolution_actions: Optional[List[str]] = None


class InspectorAction(BaseModel):
    flagged: bool = False
    flagged_fields: List[str] = Field(default_factory=list)
    issues: List[Dict[str, str]] = Field(default_factory=list)
    suggested_corrections: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class InspectorObservation(BaseModel):
    task_id: str
    task_type: str
    ticket: Ticket
    worker_response: WorkerResponse
    available_categories: List[str] = Field(default_factory=lambda: [
        "billing", "technical", "account", "product", "shipping"
    ])
    available_priorities: List[str] = Field(default_factory=lambda: [
        "low", "medium", "high", "urgent"
    ])
    available_departments: List[str] = Field(default_factory=lambda: [
        "finance", "engineering", "account_management", "product_team",
        "logistics", "billing_support", "technical_support", "general_support",
    ])
    instructions: str = ""
    step_number: int = 0
    max_steps: int = 3
    hints: List[str] = Field(default_factory=list)
    previous_score: float = 0.0


class ResetRequest(BaseModel):
    task: str = "inspection_easy"
    seed: Optional[int] = None


class ResetResponse(BaseModel):
    observation: InspectorObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    observation: InspectorObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class GraderComponentScore(BaseModel):
    name: str
    score: float = Field(ge=0.0, le=1.0)
    max_weight: float = Field(ge=0.0, le=1.0)
    details: str = ""


class GraderResult(BaseModel):
    component_scores: List[GraderComponentScore]
    overall_score: float = Field(ge=0.0, le=1.0)
    task_type: str


class InjectedError(BaseModel):
    error_type: ErrorType
    field: str
    ground_truth_value: Any
    corrupted_value: Any
    subtlety: int = Field(ge=1, le=3)
    description: str = ""
