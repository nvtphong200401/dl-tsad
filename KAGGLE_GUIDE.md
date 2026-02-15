# Training Models on Kaggle - Step-by-Step Guide

Since deep learning training is slow on CPU, you can use Kaggle's free GPU to train the SOTA models and download the trained model snapshots.

---

## 🎯 Overview

**What we'll do:**
1. Upload your code and data to Kaggle
2. Train AER and Anomaly Transformer models on GPU
3. Download the trained models
4. Use them locally for fast inference

**Benefits:**
- ✅ Free GPU access (T4 x2 GPUs)
- ✅ 30 hours/week GPU time
- ✅ Fast training (5-10 min vs 1+ hour on CPU)
- ✅ Better model performance (larger capacity)

---

## 📝 Step-by-Step Instructions

### Step 1: Prepare Your Files

**Option A: Create a Zip File (Easiest)**

On your local machine:
```bash
cd "C:\Users\nguyenphong\Downloads\study master\thesis"
zip -r best-tsad.zip best-tsad/
```

Or on Windows (PowerShell):
```powershell
Compress-Archive -Path "best-tsad" -DestinationPath "best-tsad.zip"
```

**Option B: Use Git**

Push your code to GitHub first:
```bash
cd best-tsad
git init
git add .
git commit -m "Initial commit"
git push origin main
```

---

### Step 2: Upload Data to Kaggle

1. Go to https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload your `src/data/synthetic` folder (containing all anomllm categories)
4. Name it: `anomllm-synthetic`
5. Make it **Public** or **Private**
6. Click **"Create"**

---

### Step 3: Upload Code to Kaggle

**Option A: As a Dataset (Recommended)**

1. Go to https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload `best-tsad.zip`
4. Name it: `best-tsad-code`
5. Click **"Create"**

**Option B: Direct Upload in Notebook**

You'll upload files directly in the notebook (see Step 5).

---

### Step 4: Create Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Name it: `Train Best-TSAD Models`
4. **Important:** Change Accelerator to **GPU T4 x2**
   - Click Settings (right sidebar)
   - Accelerator → GPU T4 x2
   - Save

---

### Step 5: Add Datasets to Notebook

1. In your notebook, click **"+ Add Data"** (right sidebar)
2. Search for `anomllm-synthetic` (your data)
3. Click **"Add"**
4. Search for `best-tsad-code` (your code, if uploaded)
5. Click **"Add"**

---

### Step 6: Upload the Notebook

1. Download `kaggle_train_models.ipynb` from your `best-tsad` folder
2. In Kaggle notebook, click **File → Upload Notebook**
3. Select `kaggle_train_models.ipynb`
4. Wait for upload to complete

**OR** Copy-paste cells from the notebook into your Kaggle notebook.

---

### Step 7: Update Paths in Notebook

In the notebook, update these paths:

```python
# Update this line
sys.path.insert(0, '/kaggle/working/best-tsad')

# Update this line (if you uploaded as dataset)
# If uploaded as dataset:
sys.path.insert(0, '/kaggle/input/best-tsad-code/best-tsad')

# Update data path
dataset = load_anomllm_category(CATEGORY, base_path="/kaggle/input/anomllm-synthetic/synthetic")
```

---

### Step 8: Run the Notebook

1. Click **"Run All"** or run cells one by one
2. Monitor progress (watch for errors)
3. Training will take ~5-10 minutes on GPU

**Expected output:**
```
Training AER model (device: cuda)...
  Epoch 10/50, Loss: 0.0234
  Epoch 20/50, Loss: 0.0187
  ...
AER training complete!

RESULTS
F1 Score:    0.723
PA-F1:       0.815
```

---

### Step 9: Download Trained Models

After running successfully:

1. Click **"Output"** tab (right sidebar)
2. You'll see files like:
   - `aer_model_point.pth`
   - `aer_model_point_metadata.pkl`
   - `transformer_model_point.pth`
   - `transformer_model_point_metadata.pkl`
   - `comparison_point.csv`
3. Click download icon next to each file
4. Save to your local `best-tsad/models/` folder

---

### Step 10: Use Models Locally

Create a folder for saved models:
```bash
cd best-tsad
mkdir models
```

Copy downloaded files to `models/` folder.

**Load and use the model:**

```python
import torch
import pickle
from src.models.aer import AERModel

# Load metadata
with open('models/aer_model_point_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Recreate model architecture
model = AERModel(
    input_dim=metadata['input_dim'],
    hidden_dim=metadata['hidden_dim'],
    num_layers=metadata['num_layers']
)

# Load trained weights
model.load_state_dict(torch.load('models/aer_model_point.pth', map_location='cpu'))
model.eval()

print(f"Loaded model trained on {metadata['category']}")
print(f"Performance: F1={metadata['f1']:.3f}, PA-F1={metadata['pa_f1']:.3f}")

# Now use for inference (fast!)
import numpy as np
x = torch.FloatTensor(your_data)
with torch.no_grad():
    recon, pred_f, pred_b = model(x)
```

---

## 🎓 Kaggle Tips

### GPU Time Limits

- **Free tier**: 30 hours/week GPU time
- **Timer resets**: Every Saturday
- **Save often**: Download models after each training

### Faster Training Tips

1. **Reduce data**: Train on 50K samples instead of full dataset
2. **Batch training**: Train one category at a time
3. **Save checkpoints**: Save every 10 epochs

### Common Issues

**Issue: "Out of memory"**
- Reduce `batch_size` from 32 to 16
- Reduce `hidden_dim` or `d_model`

**Issue: "Session timeout"**
- Notebook inactive for 60 min = session dies
- Stay active or download models immediately

**Issue: "Can't find module"**
- Check paths: `/kaggle/input/` vs `/kaggle/working/`
- Make sure code was uploaded correctly

---

## 📊 What to Train

### Priority Order

1. **AER on 'point' category** (best performing baseline)
2. **Transformer on 'point' category** (comparison)
3. **AER on other categories** (if you have GPU time left)

### Training Time Estimates (GPU)

| Model | Data Size | Time | GPU Memory |
|-------|-----------|------|------------|
| AER | 50K samples | ~5 min | ~2 GB |
| AER | 200K samples | ~15 min | ~4 GB |
| Transformer | 50K samples | ~3 min | ~3 GB |
| Transformer | 200K samples | ~10 min | ~6 GB |

---

## 🔄 Workflow Summary

```
1. Prepare Code & Data
   └─> Upload to Kaggle as Datasets

2. Create Notebook
   └─> Add Data
   └─> Enable GPU

3. Upload Training Notebook
   └─> Update paths
   └─> Run All

4. Training (5-10 min)
   └─> Monitor progress
   └─> Check results

5. Download Models
   └─> Save to local/models/

6. Use Locally
   └─> Fast inference with trained weights
```

---

## 💡 Alternative: Google Colab

If Kaggle doesn't work, try Google Colab (also free GPU):

1. Go to https://colab.research.google.com/
2. Upload notebook
3. Runtime → Change runtime type → GPU
4. Upload files or mount Google Drive
5. Run training
6. Download models

---

## 📁 File Structure After Download

```
best-tsad/
├── models/                           # NEW: Trained models
│   ├── aer_model_point.pth
│   ├── aer_model_point_metadata.pkl
│   ├── transformer_model_point.pth
│   └── transformer_model_point_metadata.pkl
├── src/
│   └── ... (your code)
└── kaggle_train_models.ipynb        # Notebook for Kaggle
```

---

## 🎯 Expected Results

After training on Kaggle with GPU, you should see:

| Category | Model | F1 | PA-F1 |
|----------|-------|-----|-------|
| point | AER | 0.70-0.80 | 0.60-0.70 |
| point | Transformer | 0.65-0.75 | 0.55-0.65 |

These are **much better** than baseline (F1=0.48)!

---

## ✅ Checklist

Before running on Kaggle:

- [ ] Code is in `best-tsad.zip`
- [ ] Data is uploaded as `anomllm-synthetic` dataset
- [ ] Code is uploaded as `best-tsad-code` dataset (optional)
- [ ] Notebook has GPU enabled (T4 x2)
- [ ] Paths are updated in notebook
- [ ] Ready to run!

After training:

- [ ] Downloaded `.pth` model files
- [ ] Downloaded `.pkl` metadata files
- [ ] Saved to `models/` folder locally
- [ ] Tested loading model locally
- [ ] Can run fast inference!

---

## 🚀 Next Steps

After getting trained models:

1. **Run evaluation** on all categories using loaded models
2. **Compare** AER vs Transformer vs Baseline
3. **Save results** to `src/results/synthetic/`
4. **Proceed to Phase 3** (optimization & ablation studies)

---

## 📞 Need Help?

- Kaggle Documentation: https://www.kaggle.com/docs
- Kaggle Forums: https://www.kaggle.com/discussion
- Check `PHASE2_IMPLEMENTATION.md` for troubleshooting

---

**Happy training! 🎉**
