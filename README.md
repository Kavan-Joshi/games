---
title: FleetAI - Scalable Oversight
sdk: docker
app_port: 7860
---

# FleetAI - Scalable Oversight for AI Support Agents

## Try It Now

**Hugging Face Space:** [https://huggingface.co/spaces/KavanJoshi/OpenEnv](https://huggingface.co/spaces/KavanJoshi/OpenEnv)

---

## The Problem

As companies deploy LLM-based agents for customer support, there is no standardized way to **train and evaluate oversight agents** that audit their outputs. A single misclassified ticket can cost thousands; a poorly drafted response can lose a customer. **Who watches the watchers?**

Traditional quality assurance can't scale. Every support ticket could need review, but human supervisors are expensive and slow. We need AI inspectors that can catch errors, flag policy violations, and suggest corrections — but how do you train them?

## Our Solution

FleetAI is an OpenEnv environment that creates a **two-agent ecosystem**: a Worker agent handles support tickets, and an Inspector agent reviews the Worker's output for errors, policy violations, and quality issues. The Inspector must balance catching real errors while avoiding false alarms.

```
Customer Ticket ──► Worker Agent (handles ticket)
                         │
                         ▼
                  Worker Response
                  (may contain errors)
                         │
                         ▼
                  Inspector Agent ◄── FleetAI trains this
                         │
                         ▼
              ┌──── flagged? ────┐
              │                  │
         Catches errors     Correctly ignores
         (True Positive)    clean responses
              │                  │
              ▼                  ▼
         Suggests          Avoids false
         corrections       positives
```

## Key Innovations

### 1. Error Injection with Varying Subtlety
The environment generates realistic Worker mistakes at varying difficulty levels:

| Error Type | Description | Example |
|------------|-------------|---------|
| **Obvious** | Completely wrong classification | Billing ticket classified as "technical" |
| **Subtle** | Priority off by one level | High urgency marked as "medium" |
| **Multi-error** | Two or more simultaneous errors | Wrong category + missing apology |
| **Clean traps** | Correct responses that look suspicious | Short response that's actually appropriate |

### 2. Precision-Recall Reward Model
The Inspector is scored on five independent components:

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Error Detection | 35% | Recall: did the Inspector catch actual errors? |
| Precision | 25% | Did the Inspector avoid flagging correct responses? |
| Issue Specificity | 15% | Are the stated reasons specific and evidence-based? |
| Correction Quality | 15% | Do suggested corrections match ground truth? |
| Calibration | 10% | Does confidence correlate with accuracy? |

### 3. Anti-Hack Safeguards
The reward function detects and penalizes gaming strategies:
- **Flag-everything hack**: 0.4x penalty for flagging 4+ fields when errors ≤ 1
- **Max-confidence hack**: 0.8x penalty for 0.99 confidence with wrong/no flag
- **Copy-paste hack**: 0.5x penalty for identical issue reasons across fields

### 4. Multi-Step Episodes with Progressive Hints
When the Inspector struggles, the environment provides increasingly specific hints:
- **After attempt 1**: "There are N fields with errors"
- **After attempt 2**: "Check if classification 'X' matches the ticket content"

### 5. Curriculum Learning
Training progresses through difficulty levels:
1. **Stage 1**: Easy (80 episodes) — obvious errors only
2. **Stage 2**: Hard (70 episodes) — subtle errors + clean traps  
3. **Stage 3**: Adversarial (50 episodes) — designed to trick

---

## Environment API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check and endpoint listing |
| POST | `/reset` | Initialize episode |
| POST | `/step` | Submit inspection |
| GET | `/state` | Get current observation |

### Reset Request
```json
{
  "task": "inspection_easy",
  "seed": 42
}
```

### Step Request (Inspector Action)
```json
{
  "flagged": true,
  "flagged_fields": ["priority", "response"],
  "issues": [
    {"field": "priority", "reason": "Customer said 'immediately', should be urgent not high"},
    {"field": "response", "reason": "No apology despite very_negative sentiment"}
  ],
  "suggested_corrections": {"priority": "urgent"},
  "confidence": 0.85
}
```

---

## Training Results

### Baseline vs Trained Comparison

| Difficulty | Baseline (Heuristic) | Trained (GRPO) | Delta |
|------------|---------------------|----------------|-------|
| Easy | 0.177 | 0.484 | +0.307 |
| Hard | 0.505 | 0.754 | +0.249 |
| Adversarial | 0.530 | 0.692 | +0.162 |
| **Overall** | **0.404** | **0.643** | **+0.239** |

> **Note:** Trained scores from GRPO curriculum learning on T4 GPU. Results may vary with different seeds and training configurations.

### Training Curves

#### Reward Over Time
![Training Reward](plots/training_reward.png)
*Mean reward per training step with min/max range. Upward trend indicates learning.*

#### Baseline vs Trained Performance
![Baseline vs Trained](plots/baseline_vs_trained.png)
*Comparison across all three difficulty levels. Trained model should show improvement.*

---

## Quick Start

### Option 1: Hugging Face Space
Visit the [HF Space](https://huggingface.co/spaces/KavanJoshi/OpenEnv) to interact with the environment directly.

### Option 2: Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal, run inference
export API_KEY="your-llm-api-key"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

### Option 3: Docker
```bash
docker build -t fleet-ai .
docker run -p 7860:7860 fleet-ai
```

---

## Training

### Colab Notebook (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KavanJoshi/OpenEnv/blob/main/FleetAI_Training_Colab.ipynb)

### Local Training
```bash
# Install training dependencies
pip install -e ".[train]"

# Run full training pipeline
python train.py --num_episodes 200 --num_epochs 1

# Or run specific phases
python train.py --phase data      # Generate training data
python train.py --phase sft       # SFT warm-start
python train.py --phase grpo      # GRPO RL training
python train.py --phase eval      # Evaluate baseline
python train.py --phase merge     # Merge and export model
```

### Training Pipeline
1. **Data Generation**: Curriculum-based training data with task/seed pairs
2. **SFT Warm-start**: Supervised fine-tuning on correct inspection examples
3. **GRPO Training**: Reinforcement learning with environment feedback
4. **Evaluation**: Baseline vs trained comparison across all difficulty levels

---

## Project Structure
```
.
├── server/
│   ├── app.py                    # FastAPI server with input validation
│   └── fleet_ai_environment.py   # MCP environment for OpenEnv
├── environment/
│   ├── env.py                    # FleetAIEnv: reset/step/state
│   ├── models.py                 # Pydantic schemas
│   ├── tickets.py                # 30 realistic support tickets
│   ├── graders.py                # 5-component grading + anti-hack
│   └── error_injector.py         # Worker response generator
├── tests/                        # 39 unit tests
├── FleetAI_Training_Colab.ipynb  # Colab training notebook
├── train.py                      # SFT + GRPO training script
├── evaluate.py                   # Baseline vs trained comparison
├── demo_ui.py                    # Gradio web interface
├── inference.py                  # LLM inference script
├── utils.py                      # Shared utilities
├── Dockerfile
├── openenv.yaml
└── README.md
```

---

## Additional Resources

- **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/KavanJoshi/OpenEnv)
- **Full Writeup:** [WRITEUP.md](./WRITEUP.md)
- **Video Demo:** [YouTube Link TBD]
- **Blog Post:** [Hugging Face Blog TBD]
- **Presentation:** [Slides Link TBD]

---

## Theme Alignment

**Theme 1: Multi-Agent Interactions** (Fleet AI Sub-theme)

> *Scalable Oversight: Environments that train oversight agents to monitor, analyze, and explain the behavior of other AI agents operating in complex, multi-agent settings.*

FleetAI directly addresses this by creating a training environment for inspectors who must understand and evaluate the behavior of worker agents handling customer support.

---

## Citation

If you use FleetAI in your research, please cite:

```bibtex
@misc{fleetai2024,
  title={FleetAI: Scalable Oversight for AI Support Agents},
  author={Your Name},
  year={2024},
  howpublished={\url{https://huggingface.co/spaces/KavanJoshi/OpenEnv}}
}
```

---

## License

MIT
