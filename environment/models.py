from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
from enum import Enum


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    ROUTING_RESPONSE = "routing_response"
    FULL_RESOLUTION = "full_resolution"


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    PRODUCT = "product"
    SHIPPING = "shipping"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class CustomerTier(str, Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"


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


class Observation(BaseModel):
    task_id: str
    task_type: str
    ticket: Ticket
    instructions: str
    available_categories: List[str] = Field(default_factory=lambda: [c.value for c in Category])
    available_priorities: List[str] = Field(default_factory=lambda: [p.value for p in Priority])
    available_departments: List[str] = Field(default_factory=lambda: [
        "finance", "engineering", "account_management", "product_team", "logistics",
        "general_support", "billing_support", "technical_support",
    ])
    customer_history: Optional[CustomerHistory] = None
    step_number: int = 0
    max_steps: int = 1


class Action(BaseModel):
    classification: Optional[str] = None
    priority: Optional[str] = None
    department: Optional[str] = None
    response: Optional[str] = None
    resolution_actions: Optional[List[str]] = None
    internal_notes: Optional[str] = None


class GraderComponentScore(BaseModel):
    name: str
    score: float = Field(ge=0.0, le=1.0)
    max_weight: float = Field(ge=0.0, le=1.0)
    details: str = ""


class GraderResult(BaseModel):
    component_scores: List[GraderComponentScore]
    overall_score: float = Field(ge=0.0, le=1.0)
    task_type: str


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResponse(BaseModel):
    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task: str = "classification"
    seed: Optional[int] = None
