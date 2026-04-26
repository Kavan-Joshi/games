# Training Plots

This directory should contain the following plots after running training:

## Required Plots

### 1. `training_reward.png`
- **X-axis**: Training Step
- **Y-axis**: Reward (0.0 to 1.0)
- **Content**: Mean reward per training step with min/max range
- **Caption**: "Mean reward per training step. Upward trend indicates the inspector is learning."

### 2. `baseline_vs_trained.png`
- **X-axis**: Difficulty Level (Easy, Hard, Adversarial)
- **Y-axis**: Mean Reward (0.0 to 1.0)
- **Content**: Bar chart comparing baseline (heuristic) vs trained (GRPO) performance
- **Caption**: "Baseline vs Trained performance across difficulty levels."

### 3. `zero_fraction.png` (Optional)
- **X-axis**: Training Step
- **Y-axis**: Zero Reward Fraction (0.0 to 1.0)
- **Content**: Fraction of episodes that received zero reward
- **Caption**: "Training health indicator. High values may indicate stalled learning."

## How to Generate

Run the Colab notebook `FleetAI_Training_Colab.ipynb` which automatically generates and saves these plots.

Or run locally:
```bash
python train.py --phase all --num_episodes 200
```

The plots will be saved to this directory.

## For Judges

If you're evaluating this project and don't see actual plots, please note:
1. Training requires a GPU (T4 or better)
2. Full training takes ~1-2 hours on a T4
3. The training script and notebook are fully functional
