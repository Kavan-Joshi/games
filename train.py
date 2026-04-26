"""
FleetAI GRPO Training Script - TRL + Unsloth

Run in Google Colab with T4/L4 GPU:
  1. pip install "unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git" trl openenv-core
  2. Upload training_data.json
  3. Run this script

Or run locally:
  pip install unsloth trl openenv-core
  python train.py --num_episodes 200 --num_epochs 1
"""

import os
import sys
import json
import random
import argparse
import time
from typing import Any, Dict, List, Optional

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./fleet_ai_model")
MAX_SEQ_LENGTH = 2048

CURRICULUM = [
    {"name": "stage_1_basics", "task": "inspection_easy", "episodes": 80, "description": "Obvious errors only"},
    {"name": "stage_2_mixed", "task": "inspection_hard", "episodes": 70, "description": "Subtle errors + clean traps"},
    {"name": "stage_3_adversarial", "task": "inspection_adversarial", "episodes": 50, "description": "Adversarial clean traps"},
]


def generate_training_data(num_episodes: int = 200, seed: int = 42, curriculum: bool = True) -> str:
    from environment import FleetAIEnv
    from environment.error_injector import ErrorInjector
    from environment.models import ResetRequest, InspectorAction

    env = FleetAIEnv()
    rng = random.Random(seed)

    conversations = []

    if curriculum:
        for stage in CURRICULUM:
            stage_eps = min(stage["episodes"], num_episodes)
            task = stage["task"]
            for episode in range(stage_eps):
                seed_val = seed + len(conversations)
                env._rng = random.Random(seed_val)
                env._error_injector = ErrorInjector(rng=env._rng)

                reset_result = env.reset(ResetRequest(task=task, seed=seed_val))
                observation = reset_result.observation
                info = reset_result.info

                has_errors = info.get("has_injected_errors", False)
                error_fields = info.get("injected_error_fields", [])
                ticket = observation.ticket
                worker = observation.worker_response

                prompt = _build_training_prompt(observation, task)
                correct_response = _build_correct_response(
                    ticket, worker, has_errors, error_fields, info
                )

                conversations.append({
                    "conversations": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": correct_response},
                    ],
                    "task": task,
                    "has_errors": has_errors,
                    "seed": seed_val,
                })
    else:
        env._rng = rng
        env._error_injector = ErrorInjector(rng=rng)
        tasks = ["inspection_easy", "inspection_hard", "inspection_adversarial"]
        task_weights = [0.4, 0.35, 0.25]

        for episode in range(num_episodes):
            task = rng.choices(tasks, weights=task_weights, k=1)[0]
            seed_val = seed + episode

            reset_result = env.reset(ResetRequest(task=task, seed=seed_val))
            observation = reset_result.observation
            info = reset_result.info

            has_errors = info.get("has_injected_errors", False)
            error_fields = info.get("injected_error_fields", [])
            ticket = observation.ticket
            worker = observation.worker_response

            prompt = _build_training_prompt(observation, task)
            correct_response = _build_correct_response(
                ticket, worker, has_errors, error_fields, info
            )

            conversations.append({
                "conversations": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": correct_response},
                ],
                "task": task,
                "has_errors": has_errors,
                "seed": seed_val,
            })

    data_path = "./training_data.json"
    with open(data_path, "w") as f:
        json.dump(conversations, f, indent=2)

    errors_count = sum(1 for c in conversations if c["has_errors"])
    print(f"Generated {len(conversations)} training examples -> {data_path}")
    print(f"  With errors: {errors_count}, Clean: {len(conversations) - errors_count}")
    for stage in CURRICULUM:
        stage_items = [c for c in conversations if c["task"] == stage["task"]]
        if stage_items:
            print(f"  {stage['name']}: {len(stage_items)} examples ({stage['description']})")
    return data_path


def _build_training_prompt(observation, task: str) -> str:
    ticket = observation.ticket
    worker = observation.worker_response
    instructions = observation.instructions

    return (
        f"You are an AI oversight inspector. Review the support worker's response.\n\n"
        f"TICKET:\n"
        f"- ID: {ticket.id}\n"
        f"- Subject: {ticket.subject}\n"
        f"- Body: {ticket.body}\n"
        f"- Customer: {ticket.customer_name} (Tier: {ticket.customer_tier})\n"
        f"- Sentiment: {ticket.sentiment}\n"
        f"- Previous Tickets: {ticket.previous_ticket_count}\n\n"
        f"WORKER'S RESPONSE:\n"
        f"- Classification: {worker.classification or 'N/A'}\n"
        f"- Priority: {worker.priority or 'N/A'}\n"
        f"- Department: {worker.department or 'N/A'}\n"
        f"- Response: {(worker.response or 'N/A')[:300]}\n"
        f"- Resolution Actions: {worker.resolution_actions or 'N/A'}\n\n"
        f"INSTRUCTIONS:\n{instructions}\n\n"
        f"Respond with a JSON object with: flagged, flagged_fields, issues, suggested_corrections, confidence"
    )


def _build_correct_response(ticket, worker, has_errors, error_fields, info) -> str:
    import json as j

    if not has_errors:
        return j.dumps({
            "flagged": False,
            "flagged_fields": [],
            "issues": [],
            "suggested_corrections": {},
            "confidence": 0.15,
        })

    issues = []
    corrections = {}
    gt = info.get("ground_truth", {})

    w_class = (worker.classification or "").lower()
    w_priority = (worker.priority or "").lower()
    w_dept = (worker.department or "").lower()
    w_resp = (worker.response or "").lower()

    if "classification" in error_fields:
        issues.append({"field": "classification", "reason": f"Ticket should be '{gt.get('category')}' not '{w_class}' based on ticket content about {ticket.subject.lower()}"})
        corrections["classification"] = gt.get("category")

    if "priority" in error_fields:
        issues.append({"field": "priority", "reason": f"Priority should be '{gt.get('priority')}' not '{w_priority}' given the customer's {ticket.sentiment} sentiment and urgency"})
        corrections["priority"] = gt.get("priority")

    if "department" in error_fields:
        issues.append({"field": "department", "reason": f"Should be routed to '{gt.get('department')}' not '{w_dept}'"})
        corrections["department"] = gt.get("department")

    if "response" in error_fields:
        if ticket.sentiment in ("negative", "very_negative"):
            has_apology = any(w in w_resp for w in ["sorry", "apologize", "apologies"])
            if not has_apology:
                issues.append({"field": "response", "reason": f"Customer has {ticket.sentiment} sentiment but response lacks apology"})
        has_name = ticket.customer_name.lower().split()[-1] in w_resp if ticket.customer_name else False
        if not has_name:
            issues.append({"field": "response", "reason": f"Response not personalized for customer '{ticket.customer_name}'"})

    if "resolution_actions" in error_fields:
        issues.append({"field": "resolution_actions", "reason": "Expected resolution actions are missing"})

    return j.dumps({
        "flagged": True,
        "flagged_fields": list(error_fields),
        "issues": issues,
        "suggested_corrections": corrections,
        "confidence": 0.85,
    })


def _parse_completion_to_action(completion: str):
    from environment.models import InspectorAction
    import re

    text = completion.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    parsed = json.loads(text)
    return InspectorAction(
        flagged=bool(parsed.get("flagged", False)),
        flagged_fields=parsed.get("flagged_fields", []),
        issues=parsed.get("issues", []),
        suggested_corrections=parsed.get("suggested_corrections", {}),
        confidence=float(parsed.get("confidence", 0.5)),
    )


def _compute_reward_for_sample(completion: str, task: str, seed: int) -> float:
    from environment import FleetAIEnv
    from environment.models import ResetRequest

    env = FleetAIEnv()
    try:
        action = _parse_completion_to_action(completion)
        env.reset(ResetRequest(task=task, seed=seed))
        step_result = env.step(action)
        return step_result.reward
    except Exception:
        return 0.0


def _make_reward_fn(dataset_tasks: List[str], dataset_seeds: List[int]):
    _reward_log = {"call_count": 0, "rewards_history": [], "generation_samples": []}

    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for i, completion in enumerate(completions):
            task = dataset_tasks[i] if i < len(dataset_tasks) else "inspection_easy"
            seed = dataset_seeds[i] if i < len(dataset_seeds) else 42
            rewards.append(_compute_reward_for_sample(completion, task, seed))
        _reward_log["call_count"] += 1

        entry = {
            "call": _reward_log["call_count"],
            "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
            "min_reward": round(min(rewards), 4) if rewards else 0.0,
            "max_reward": round(max(rewards), 4) if rewards else 0.0,
            "zero_fraction": round(sum(1 for r in rewards if r == 0.0) / len(rewards), 4) if rewards else 0.0,
        }
        _reward_log["rewards_history"].append(entry)

        if _reward_log["call_count"] % 10 == 0 and completions:
            _reward_log["generation_samples"].append({
                "call": _reward_log["call_count"],
                "sample_completion": completions[0][:500],
                "sample_reward": rewards[0] if rewards else 0.0,
                "sample_task": dataset_tasks[0] if dataset_tasks else "unknown",
            })

        return rewards
    return reward_fn


def run_grpo_training(data_path: str, output_dir: str, num_epochs: int = 1):
    """
    GRPO training using TRL + Unsloth.

    This function requires:
      pip install unsloth trl openenv-core
    """
    import torch
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    print("\n" + "=" * 60)
    print("  LOADING MODEL")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("  LOADING DATA")
    print("=" * 60)

    with open(data_path) as f:
        data = json.load(f)

    dataset_tasks = [item.get("task", "inspection_easy") for item in data]
    dataset_seeds = [item.get("seed", 42) for item in data]

    grpo_data = []
    for i, item in enumerate(data):
        grpo_data.append({
            "prompt": item["conversations"][0]["content"],
            "task": item.get("task", "inspection_easy"),
            "seed": item.get("seed", 42),
        })

    dataset = Dataset.from_list(grpo_data)
    dataset_tasks = [item["task"] for item in grpo_data]
    dataset_seeds = [item["seed"] for item in grpo_data]

    reward_fn = _make_reward_fn(dataset_tasks, dataset_seeds)

    dataset = dataset.sort("task")

    print("\n" + "=" * 60)
    print("  GRPO TRAINING")
    print("=" * 60)
    print(f"  Dataset sorted by task for curriculum progression:")
    task_counts = {}
    for item in data:
        t = item.get("task", "unknown")
        task_counts[t] = task_counts.get(t, 0) + 1
    for t, c in task_counts.items():
        print(f"    {t}: {c} examples")

    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        optim="adamw_8bit",
        max_completion_length=256,
        num_generations=2,
        temperature=0.7,
        report_to=os.environ.get("FLEET_REPORT_TO", "tensorboard"),
        run_name=f"fleet-ai-grpo-{int(time.time())}",
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=config,
        reward_funcs=reward_fn,
    )

    print("Starting training...")
    train_result = trainer.train()

    print("\n" + "=" * 60)
    print("  TRAINING METRICS")
    print("=" * 60)
    metrics = train_result.metrics
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    generation_log_path = os.path.join(output_dir, "generation_log.json")
    try:
        log_data = reward_fn.__closure__[0].cell_contents
        with open(generation_log_path, "w") as f:
            json.dump({
                "rewards_history": log_data["rewards_history"],
                "generation_samples": log_data["generation_samples"],
            }, f, indent=2)
        print(f"  Generation reward log saved to {generation_log_path}")
        print(f"  Generation text samples: {len(log_data['generation_samples'])}")
    except Exception:
        print("  Could not save generation reward log")

    print("\n" + "=" * 60)
    print("  SAVING MODEL")
    print("=" * 60)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Model saved to {output_dir}")

    print("\n  To push to HuggingFace:")
    print(f"    model.push_to_hub('your-username/fleet-ai-inspector')")
    print(f"    tokenizer.push_to_hub('your-username/fleet-ai-inspector')")


def run_sft_training(data_path: str, output_dir: str, num_epochs: int = 1):
    """
    SFT warm-start training using TRL + Unsloth.
    Run this BEFORE GRPO for best results.
    """
    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from datasets import Dataset
    from transformers import TrainingArguments

    print("\n" + "=" * 60)
    print("  SFT WARM-START TRAINING")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    torch.cuda.empty_cache()

    with open(data_path) as f:
        data = json.load(f)

    sft_examples = []
    for item in data:
        convs = item["conversations"]
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{convs[0]['content']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{convs[1]['content']}<|eot_id|>"
        sft_examples.append({"text": text})

    dataset = Dataset.from_list(sft_examples)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        args=TrainingArguments(
            output_dir=f"{output_dir}_sft",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=num_epochs,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            optim="adamw_8bit",
            report_to=os.environ.get("FLEET_REPORT_TO", "tensorboard"),
            run_name=f"fleet-ai-sft-{int(time.time())}",
        ),
    )

    trainer.train()

    model.save_pretrained(f"{output_dir}_sft")
    tokenizer.save_pretrained(f"{output_dir}_sft")
    print(f"  SFT model saved to {output_dir}_sft")


def merge_and_export_model(adapter_dir: str, output_dir: str, model_name: str = MODEL_NAME):
    """
    Safely merge LoRA adapters into the base model and save as 16-bit.
    Avoids the 4-bit -> 16-bit upcast corruption the hackathon guide warns about.
    """
    import torch
    from unsloth import FastLanguageModel

    print("\n" + "=" * 60)
    print("  MERGING & EXPORTING MODEL")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )

    model.load_adapter(adapter_dir)

    model = model.merge_and_unload()

    merged_dir = f"{output_dir}_merged"
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"  Merged 16-bit model saved to {merged_dir}")

    print("\n  To push the merged model to HuggingFace:")
    print(f"    from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"    m = AutoModelForCausalLM.from_pretrained('{merged_dir}')")
    print(f"    t = AutoTokenizer.from_pretrained('{merged_dir}')")
    print(f"    m.push_to_hub('your-username/fleet-ai-inspector-merged')")
    print(f"    t.push_to_hub('your-username/fleet-ai-inspector-merged')")

    return merged_dir


def evaluate_model(model_dir: str, num_tickets: int = 10) -> Dict[str, Any]:
    from environment import FleetAIEnv
    from environment.models import ResetRequest

    print(f"\n  Evaluating model: {model_dir}")
    print("  (Using heuristic inspector — replace with LLM inference for trained model)")

    env = FleetAIEnv()
    results = {}

    for task in ["inspection_easy", "inspection_hard", "inspection_adversarial"]:
        scores = []
        for seed in range(num_tickets):
            r = env.reset(ResetRequest(task=task, seed=seed))
            observation = r.observation
            ticket = observation.ticket
            worker = observation.worker_response

            action = _build_heuristic_action(ticket, worker)
            result = env.step(action)
            scores.append(result.reward)

        results[task] = {
            "mean": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "scores": scores,
        }
        print(f"  {task:30s}: mean={results[task]['mean']:.3f} "
              f"range=[{results[task]['min']:.3f}, {results[task]['max']:.3f}]")

    return results


def evaluate_baseline(num_tickets: int = 10) -> Dict[str, Any]:
    return evaluate_model("baseline", num_tickets=num_tickets)


def _build_heuristic_action(ticket, worker) -> "InspectorAction":
    from environment.models import InspectorAction

    flagged = False
    flagged_fields = []
    issues = []
    suggested = {}

    sentiment = (ticket.sentiment or "neutral").lower()
    w_resp = (worker.response or "").lower()
    w_class = (worker.classification or "").lower()
    w_priority = (worker.priority or "").lower()
    w_dept = (worker.department or "").lower()
    w_actions = worker.resolution_actions or []

    if sentiment in ("negative", "very_negative"):
        has_apology = any(w in w_resp for w in ["sorry", "apologize", "apologies", "regret", "inconvenience"])
        if not has_apology:
            flagged = True
            flagged_fields.append("response")
            issues.append({"field": "response", "reason": f"Customer sentiment is {sentiment} but no apology in response"})
            suggested["response"] = f"Dear {ticket.customer_name}, I sincerely apologize..."

    body_lower = f"{(ticket.subject or '').lower()} {(ticket.body or '').lower()}"

    urgent_keywords = ["urgent", "immediately", "asap", "emergency", "critical", "production"]
    has_urgency = any(kw in body_lower for kw in urgent_keywords)
    if has_urgency and w_priority in ("low", "medium"):
        flagged = True
        flagged_fields.append("priority")
        issues.append({"field": "priority", "reason": "Ticket contains urgency language but priority is too low"})
        suggested["priority"] = "urgent" if sentiment == "very_negative" else "high"

    customer_name_last = (ticket.customer_name or "").split()[-1].lower() if ticket.customer_name else ""
    if customer_name_last and customer_name_last not in w_resp:
        flagged = True
        flagged_fields.append("response")
        issues.append({"field": "response", "reason": f"Response not personalized for customer '{ticket.customer_name}'"})

    if not w_actions or len(w_actions) == 0:
        flagged = True
        flagged_fields.append("resolution_actions")
        issues.append({"field": "resolution_actions", "reason": "No resolution actions provided"})

    if not flagged:
        flagged = False
        flagged_fields = []
        issues = []
        suggested = {}

    return InspectorAction(
        flagged=flagged,
        flagged_fields=flagged_fields,
        issues=issues,
        suggested_corrections=suggested,
        confidence=0.6 if flagged else 0.2,
    )


def main():
    parser = argparse.ArgumentParser(description="FleetAI Training Pipeline")
    parser.add_argument("--phase", choices=["data", "sft", "grpo", "eval", "merge", "all"],
                        default="all", help="Training phase to run")
    parser.add_argument("--num_episodes", type=int, default=200,
                        help="Number of training episodes to generate")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="Base model to train")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum data generation")
    args = parser.parse_args()

    p = lambda msg: print(msg, flush=True)

    p("=" * 60)
    p("  FLEET AI - TRAINING PIPELINE")
    p("=" * 60)
    p(f"  Model: {args.model}")
    p(f"  Phase: {args.phase}")
    p(f"  Episodes: {args.num_episodes}")
    p(f"  Epochs: {args.num_epochs}")
    p(f"  Curriculum: {'disabled' if args.no_curriculum else 'enabled'}")
    p("=" * 60)

    if args.phase in ("data", "all"):
        p("\n[Phase 1/5] Generating training data...")
        data_path = generate_training_data(
            num_episodes=args.num_episodes,
            curriculum=not args.no_curriculum,
        )
    else:
        data_path = "./training_data.json"

    if args.phase in ("eval", "all"):
        p("\n[Phase 2/5] Evaluating baseline...")
        baseline = evaluate_baseline()
        baseline_path = "./baseline_results.json"
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)
        p(f"  Baseline saved to {baseline_path}")

    if args.phase in ("sft", "all"):
        p("\n[Phase 3/5] SFT warm-start training...")
        try:
            run_sft_training(data_path, f"{OUTPUT_DIR}_sft", args.num_epochs)
        except ImportError as e:
            p(f"  Skipping SFT: {e}")
            p("  Install with: pip install unsloth trl")

    if args.phase in ("grpo", "all"):
        p("\n[Phase 4/5] GRPO RL training...")
        try:
            model_path = f"{OUTPUT_DIR}_sft" if os.path.exists(f"{OUTPUT_DIR}_sft/config.json") else args.model
            run_grpo_training(data_path, OUTPUT_DIR, args.num_epochs)
        except ImportError as e:
            p(f"  Skipping GRPO: {e}")
            p("  Install with: pip install unsloth trl")

    if args.phase in ("merge", "all"):
        p("\n[Phase 5/5] Merging & exporting model...")
        adapter_dir = OUTPUT_DIR if os.path.exists(OUTPUT_DIR) else f"{OUTPUT_DIR}_sft"
        if os.path.exists(adapter_dir):
            try:
                merge_and_export_model(adapter_dir, OUTPUT_DIR, model_name=args.model)
            except ImportError as e:
                p(f"  Skipping merge: {e}")
                p("  Install with: pip install unsloth")
        else:
            p(f"  No adapter found at {adapter_dir}, skipping merge")

    p("\n" + "=" * 60)
    p("  NEXT STEPS")
    p("=" * 60)
    p("  1. Push trained model: model.push_to_hub('user/fleet-ai')")
    p("  2. Update inference.py MODEL_NAME to your trained model")
    p("  3. Run evaluation: python train.py --phase eval")
    p("  4. Compare baseline vs trained reward curves")
    p("  5. Run demo: python demo.py")
    p("=" * 60)


if __name__ == "__main__":
    main()
