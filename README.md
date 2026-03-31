# Customer Support Triage - OpenEnv Environment

A real-world customer support ticket triage and resolution environment for evaluating LLM agents on genuine support tasks.

## Overview

This environment simulates the daily work of customer support agents: reading incoming tickets, classifying them, routing to the right team, drafting professional responses, and recommending resolution actions. Every task in this environment is something human support agents do daily at companies of all sizes.

**Why this matters:** Companies spend $300B+ annually on customer support. LLM-based agents are actively being deployed to triage and resolve tickets. This environment provides a standardized benchmark for evaluating how well agents handle this real-world task.

## Tasks

### Task 1: Ticket Classification (Easy)

Classify a support ticket into the correct category and assign the appropriate priority level.

| Component | Weight | Description |
|-----------|--------|-------------|
| Classification | 60% | Correct category from [billing, technical, account, product, shipping] |
| Priority | 40% | Correct priority from [low, medium, high, urgent] |

**Scoring:** Exact match = 1.0, alias/close match = 0.3-0.5, wrong = 0.0

### Task 2: Routing and Response Drafting (Medium)

Classify, route to the correct department, and draft a professional customer-facing response.

| Component | Weight | Description |
|-----------|--------|-------------|
| Classification | 15% | Correct category |
| Priority | 10% | Correct priority |
| Department | 20% | Correct department routing |
| Response | 55% | Professional response quality |

**Response grading checks:** key term coverage (30%), greeting (8%), apology when needed (10%), actionable content (20%), appropriate length (7%), personalization (20%)

### Task 3: Full Ticket Resolution (Hard)

Complete resolution with customer history analysis, contextual response, and resolution plan.

| Component | Weight | Description |
|-----------|--------|-------------|
| Classification | 10% | Correct category |
| Priority | 5% | Correct priority |
| Department | 10% | Correct department |
| Response | 40% | Context-aware professional response |
| Resolution | 35% | Appropriate resolution actions |

**Context usage:** Agent must reference customer loyalty, recent issues, escalation history, and lifetime value when relevant.

## Action Space

```json
{
  "classification": "string | null",
  "priority": "string | null",
  "department": "string | null",
  "response": "string | null",
  "resolution_actions": ["string"] | null
}
```

| Field | Required for | Values |
|-------|-------------|--------|
| classification | all tasks | billing, technical, account, product, shipping |
| priority | all tasks | low, medium, high, urgent |
| department | routing_response, full_resolution | finance, engineering, account_management, product_team, logistics, billing_support, technical_support, general_support |
| response | routing_response, full_resolution | Free text (professional customer-facing response) |
| resolution_actions | full_resolution | List of action strings |

## Observation Space

```json
{
  "task_id": "TKT-001",
  "task_type": "classification",
  "ticket": {
    "id": "TKT-001",
    "subject": "...",
    "body": "...",
    "customer_id": "CUS-101",
    "customer_name": "...",
    "customer_tier": "gold",
    "customer_tenure_days": 540,
    "sentiment": "negative",
    "previous_ticket_count": 3,
    "created_at": "..."
  },
  "instructions": "TASK: Classify the support ticket...",
  "available_categories": ["billing", "technical", "account", "product", "shipping"],
  "available_priorities": ["low", "medium", "high", "urgent"],
  "available_departments": ["finance", "engineering", ...],
  "customer_history": { ... },
  "step_number": 0,
  "max_steps": 1
}
```

The `customer_history` field is only populated for the `full_resolution` task.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check and endpoint listing |
| POST | `/reset` | Initialize episode. Body: `{"task": "classification", "seed": 42}` |
| POST | `/step` | Submit action. Body: Action JSON |
| GET | `/state` | Get current observation |

## Setup

### Local Development

```bash
pip install -r requirements.txt
python app.py
```

### Docker

```bash
docker build -t customer-support-triage .
docker run -p 7860:7860 customer-support-triage
```

### Running Baseline Inference

```bash
export OPENAI_API_KEY="your-key-here"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

## Baseline Scores

| Task | gpt-4o-mini | Description |
|------|-------------|-------------|
| classification | ~0.85 | Strong classification, occasional priority mismatches |
| routing_response | ~0.55 | Good classification, variable response quality |
| full_resolution | ~0.40 | Context awareness is challenging |

*Scores are approximate and may vary. Run `inference.py` with your API key for exact results.*

## Grading System

All graders are deterministic and produce scores in [0.0, 1.0]:

- **Classification:** Exact string match (1.0), alias resolution (0.3-0.5), wrong (0.0)
- **Priority:** Exact match (1.0), one level off (0.5), two levels off (0.2), wrong (0.0)
- **Department:** Exact match (1.0), alias match (0.7), related (0.5), wrong (0.0)
- **Response:** Composite heuristic with weighted sub-scores
- **Resolution:** Action matching + escalation check + tier awareness

## Project Structure

```
.
├── openenv.yaml          # OpenEnv metadata and task definitions
├── Dockerfile            # Container definition for HF Spaces
├── app.py                # FastAPI server
├── inference.py          # Baseline inference script (uses OpenAI API)
├── requirements.txt      # Python dependencies
├── environment/
│   ├── __init__.py       # Package exports
│   ├── models.py         # Pydantic models (Observation, Action, StepResponse)
│   ├── env.py            # CustomerSupportEnv (step/reset/state)
│   ├── tickets.py        # 30 realistic tickets with ground truth labels
│   └── graders.py        # Deterministic grading functions
└── README.md
```

## License

MIT
