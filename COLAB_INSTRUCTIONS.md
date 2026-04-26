# How to Run FleetAI Training in Google Colab

## Step 1: Open Google Colab

1. Go to https://colab.research.google.com/
2. Click **"File" → "Upload notebook"**
3. Upload `FleetAI_Training_Colab.ipynb` from this directory

## Step 2: Enable GPU

1. Click **"Runtime" → "Change runtime type"**
2. Set **"Hardware accelerator"** to **"T4 GPU"**
3. Click **"Save"**

## Step 3: Upload Environment Files

Before running the notebook, you need to upload these files:

**Option A: Upload via file browser**
1. Click the **folder icon** on the left sidebar
2. Click the **upload icon** (↑)
3. Create a folder called `environment`
4. Upload these files:
   - `environment/__init__.py`
   - `environment/env.py`
   - `environment/models.py`
   - `environment/tickets.py`
   - `environment/graders.py`
   - `environment/error_injector.py`
5. Upload `train.py` to the root
6. Upload `utils.py` to the root

**Option B: Clone from HuggingFace (after you push)**
```python
# Run this in a Colab cell
!git clone https://huggingface.co/spaces/YOUR_USERNAME/fleet-ai
%cd fleet-ai
```

## Step 4: Run the Notebook

Run cells in order:
1. **Cell 1**: Install Dependencies (~2 min)
2. **Cell 2**: GPU Check (should show T4)
3. **Cell 3**: Clone FleetAI (skip if uploaded manually)
4. **Cell 4**: Configuration (adjust if needed)
5. **Cell 5**: Generate Training Data (~30 sec)
6. **Cell 6**: Evaluate Baseline (~1 min)
7. **Cell 7**: SFT Training (~15-30 min)
8. **Cell 8**: GRPO Training (~30-60 min)
9. **Cell 9-10**: Evaluate Trained Model
10. **Cell 11-12**: Generate Plots
11. **Cell 13-14**: Save and Download Results

## Step 5: Download Results

After training completes:
1. Run Cell 14 to download the results ZIP
2. Extract it locally
3. Copy `plots/*.png` files to your repo's `plots/` folder
4. Update README with the results

## Estimated Time

| Phase | Duration |
|-------|----------|
| Setup | 5 min |
| Data Generation | 1 min |
| Baseline Evaluation | 2 min |
| SFT Training | 15-30 min |
| GRPO Training | 30-60 min |
| Evaluation + Plots | 5 min |
| **Total** | **~1-1.5 hours** |

## Troubleshooting

### "CUDA out of memory"
- Reduce `BATCH_SIZE` to 1
- Reduce `NUM_EPISODES` to 100

### "Module not found"
- Make sure you uploaded all `environment/*.py` files
- Run `%cd fleet-ai` if using git clone

### "GPU not available"
- Make sure you selected T4 GPU in runtime settings
- Try Runtime → Disconnect and delete runtime, then reconnect

## Quick Start (Alternative)

If you just want to test that training works without full training:

1. Set `NUM_EPISODES = 50` in Cell 4
2. Skip Cell 7 (SFT Training)
3. This reduces total time to ~15 minutes
