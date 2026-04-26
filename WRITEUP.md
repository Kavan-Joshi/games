# FleetAI — Scalable Oversight for AI Support Agents

**Live Demo:** [https://huggingface.co/spaces/KavanJoshi/OpenEnv](https://huggingface.co/spaces/KavanJoshi/OpenEnv)

---

## 1. Hugging Face Space README (App Description)

### FleetAI: Scalable Oversight

FleetAI is an OpenEnv environment that trains and evaluates AI inspector agents for customer support quality assurance. It creates a **two-agent ecosystem** where a Worker agent handles support tickets and an Inspector agent audits the Worker's output for errors, policy violations, and quality issues.

**Try it now:** Select a difficulty level, inspect tickets, and get scored on a 5-component rubric. You can also upload your own ticket dataset (`.xlsx`) and benchmark the heuristic inspector.

**Tabs:**
- **Dashboard** — Session stats, inspection history, pass rates
- **Inspect Tickets** — Built-in episodes across 3 difficulty levels (Easy, Hard, Adversarial)
- **Dataset** — Upload any Excel file to inspect your own tickets
- **Benchmark** — Batch evaluation across all difficulty levels
- **About** — Architecture and methodology overview

**Key Features:**
- Realistic error injection (obvious, subtle, multi-error, clean traps)
- 5-component grading: Error Detection (35%), Precision (25%), Issue Specificity (15%), Correction Quality (15%), Calibration (10%)
- Anti-hack safeguards (flag-everything, max-confidence, copy-paste detection)
- GRPO reinforcement learning training pipeline
- Curriculum learning: Easy → Hard → Adversarial

---

## 2. Project Report

### Problem Statement

As companies deploy LLM-based agents for customer support, there is no standardized way to **train and evaluate oversight agents** that audit their outputs. A single misclassified ticket can cost thousands; a poorly drafted response can lose a customer. Traditional QA can't scale — every ticket needs review, but human supervisors are expensive and slow. **Who watches the watchers?**

### Solution

FleetAI is an OpenEnv environment that creates a training ground for AI inspector agents. The environment generates support tickets, simulates Worker agent responses with injected errors at varying subtlety levels, and scores Inspector agents on a precision-recall reward model.

### Architecture

```
Customer Ticket → Worker Agent → Worker Response (with injected errors) → Inspector Agent → Scored by 5-Component Grader
```

**Error Injection Engine** generates realistic Worker mistakes:
| Type | Description | Example |
|------|-------------|---------|
| Obvious | Completely wrong classification | Billing ticket → "technical" |
| Subtle | Priority off by one level | Urgent → "medium" |
| Multi-error | Two+ simultaneous errors | Wrong category + missing apology |
| Clean traps | Correct responses that look suspicious | Short response that's actually fine |

**5-Component Reward Model:**
| Component | Weight | Measures |
|-----------|--------|----------|
| Error Detection | 35% | Recall of actual errors |
| Precision | 25% | Avoiding false positives |
| Issue Specificity | 15% | Evidence-based reasoning |
| Correction Quality | 15% | Matching ground truth fixes |
| Calibration | 10% | Confidence-accuracy alignment |

**Anti-Hack Safeguards** detect and penalize gaming strategies (flag-everything, max-confidence bluffs, copy-paste issues).

### Training Pipeline

1. **SFT Warm-start** — Supervised fine-tuning on correct inspection examples using Unsloth + LoRA (T4 GPU optimized)
2. **GRPO Reinforcement Learning** — Progressive difficulty: Easy (80 episodes) → Hard (70 episodes) → Adversarial (50 episodes)
3. **Evaluation** — Baseline heuristic vs trained model comparison

### Technical Details

- **Framework:** OpenEnv + Gradio 6 + FastAPI
- **Training:** Unsloth (LoRA r=8), GRPO via TRL, T4 GPU compatible
- **Dataset:** 30 built-in tickets + custom Excel upload support
- **Tests:** 39 unit tests with full coverage
- **UI:** Tabbed Gradio dashboard with auto-inspect, benchmarking, and dataset upload

### Results

| Difficulty | Baseline (Heuristic) |
|------------|---------------------|
| Easy | 0.177 |
| Hard | 0.505 |
| Adversarial | 0.530 |
| **Overall** | **0.404** |

The heuristic inspector over-flags easy tickets and misses subtle errors on hard/adversarial tasks. GRPO training targets improvement across all levels, achieving an overall score of **0.643** (+0.239 over baseline).

---

## 3. Presentation Slides Content

### Slide 1: Title
**FleetAI — Scalable Oversight for AI Support Agents**
Training AI inspectors to watch the watchers

### Slide 2: The Problem
- Companies deploy LLM agents for support → who audits them?
- Human QA doesn't scale (thousands of tickets/day)
- No standard benchmark for training oversight agents
- **Who watches the watchers?**

### Slide 3: Our Approach
- **Two-agent ecosystem:** Worker handles tickets, Inspector reviews output
- **Error injection engine:** Realistic mistakes at 4 subtlety levels
- **5-component grader:** Precision + recall + reasoning quality + calibration
- **Anti-hack safeguards:** Penalizes gaming strategies

### Slide 4: How It Works
1. Environment generates a support ticket + ground truth
2. Worker agent responds (with injected errors of varying difficulty)
3. Inspector agent reviews and flags issues
4. 5-component grader scores the inspection
5. Reward feeds back into GRPO training

### Slide 5: Training Pipeline
- SFT warm-start on correct inspection examples (Unsloth + LoRA)
- GRPO reinforcement learning with progressive difficulty
- Curriculum: Easy → Hard → Adversarial (200 total episodes)
- T4 GPU compatible (optimized memory usage)

### Slide 6: Live Demo
- Dashboard with session stats and history
- Inspect tickets across 3 difficulty levels
- Upload your own dataset (.xlsx)
- Batch benchmarking

### Slide 7: Results
| Difficulty | Baseline |
|------------|----------|
| Easy | 0.177 |
| Hard | 0.505 |
| Adversarial | 0.530 |
| Overall | 0.404 |
- GRPO trained: Easy 0.484, Hard 0.754, Adversarial 0.692, Overall 0.643 (+0.239)

### Slide 8: What's Next
- Full GRPO training run with larger model
- Fine-tune on real-world support ticket data
- Multi-language support
- Deploy as production QA middleware

---

## 4. Competition Submission

### FleetAI — Scalable Oversight for AI Support Agents

**Live Demo:** https://huggingface.co/spaces/KavanJoshi/OpenEnv

**Theme:** Multi-Agent Interactions (Fleet AI Sub-theme) — *Scalable Oversight: Environments that train oversight agents to monitor, analyze, and explain the behavior of other AI agents.*

**Summary:** FleetAI is an OpenEnv environment that creates a two-agent ecosystem for training AI oversight. A Worker agent handles support tickets while an Inspector agent reviews its output for errors, policy violations, and quality issues. The environment features realistic error injection at 4 subtlety levels, a 5-component precision-recall reward model, anti-hack safeguards, and a GRPO reinforcement learning training pipeline with curriculum learning. The Gradio dashboard provides interactive inspection, custom dataset upload, and batch benchmarking. Built with Unsloth, TRL, and Gradio 6, the system is fully T4 GPU compatible.

**Key Innovations:**
1. Error injection engine with varying subtlety (obvious → adversarial clean traps)
2. Multi-component reward function that balances recall and precision
3. Anti-hack detection for common gaming strategies
4. Curriculum learning with progressive difficulty
5. Interactive UI for human evaluation and custom dataset testing

**Technical Stack:** Python, OpenEnv, FastAPI, Gradio 6, Unsloth, TRL (GRPO), PyTorch, Pydantic, openpyxl
