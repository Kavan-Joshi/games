# Training Results Template

After running training, fill in this document with your results.

## Model Configuration

- **Base Model**: meta-llama/Llama-3.1-8B-Instruct
- **Training Method**: GRPO (Group Relative Policy Optimization)
- **Episodes**: 200
- **Epochs**: 1
- **Curriculum**: Easy → Hard → Adversarial

## Baseline Results (Heuristic Inspector)

| Task | Mean Reward | Min | Max |
|------|-------------|-----|-----|
| inspection_easy | TBD | TBD | TBD |
| inspection_hard | TBD | TBD | TBD |
| inspection_adversarial | TBD | TBD | TBD |

## Trained Results (GRPO)

| Task | Mean Reward | Min | Max |
|------|-------------|-----|-----|
| inspection_easy | TBD | TBD | TBD |
| inspection_hard | TBD | TBD | TBD |
| inspection_adversarial | TBD | TBD | TBD |

## Improvement Summary

| Task | Baseline | Trained | Δ (Improvement) |
|------|----------|---------|-----------------|
| inspection_easy | TBD | TBD | TBD |
| inspection_hard | TBD | TBD | TBD |
| inspection_adversarial | TBD | TBD | TBD |
| **Overall** | **TBD** | **TBD** | **TBD** |

## Key Observations

1. _Fill in after training_
2. _Example: "Trained model shows 2x improvement on adversarial tasks"_
3. _Example: "Zero reward fraction decreased from 40% to 15%"_

## Training Health Metrics

- **Zero reward episodes**: TBD%
- **Mean reward at start**: TBD
- **Mean reward at end**: TBD
- **Training time**: TBD minutes

## Plots Generated

- [ ] `plots/training_reward.png` - Reward over training steps
- [ ] `plots/baseline_vs_trained.png` - Performance comparison
- [ ] `plots/zero_fraction.png` - Training health indicator
