# Training Scripts

This directory contains scripts for training machine learning models.

## Available Scripts

### 1. `train_symptom_model.py`
Trains the symptom-based disease prediction model using Random Forest.

**Usage:**
```bash
python scripts/train_symptom_model.py
```

**Output:**
- `backend/models/symptom_model.pkl`
- `backend/models/symptom_encoder.pkl`
- `backend/models/label_encoder.pkl`

---

### 2. `train_xray_model_real.py` ⭐ **NEW**
Trains a real X-ray classification model using ResNet50 transfer learning.

**Prerequisites:**
1. Install TensorFlow: `pip install tensorflow==2.15.0`
2. Download and organize X-ray dataset (see `data/xray_dataset/SETUP_GUIDE.md`)

**Usage:**
```bash
python scripts/train_xray_model_real.py
```

**What it does:**
- Checks dataset organization
- Creates data generators with augmentation
- Builds ResNet50-based CNN model
- Trains in 2 phases (frozen base → fine-tuning)
- Saves best model and training plots

**Output:**
- `backend/models/xray_model.h5` (main model)
- `backend/models/xray_model_phase1.h5` (checkpoint)
- `backend/models/training_history.png` (plots)

**Training time:**
- GPU: 30-60 minutes
- CPU: 3-6 hours

**Expected accuracy:** 85-95% (with proper dataset)

---

### 3. `data_preprocessing.py`
Utility functions for data preprocessing (used by training scripts).

---

## Quick Start: Train X-Ray Model

See **`XRAY_TRAINING_GUIDE.md`** in the project root for complete step-by-step instructions!

**TL;DR:**
```bash
# 1. Install dependencies
pip install tensorflow==2.15.0 matplotlib

# 2. Download dataset (see SETUP_GUIDE.md)
cd data/xray_dataset
# ... download from Kaggle ...

# 3. Train model
python scripts/train_xray_model_real.py
```

---

## Model Comparison

| Model | Type | Input | Output | Accuracy |
|-------|------|-------|--------|----------|
| Symptom Model | Random Forest | Symptoms list | Disease predictions | ~80% |
| X-Ray Model | CNN (ResNet50) | X-ray image | Condition predictions | 85-95% |

---

## Notes

- All models are saved to `backend/models/`
- The Flask backend automatically loads trained models
- Demo mode is used when models are not available
- For production use, retrain with larger, validated datasets

## Need Help?

- **X-Ray Model**: See `XRAY_TRAINING_GUIDE.md`
- **Dataset Setup**: See `data/xray_dataset/SETUP_GUIDE.md`
- **General Info**: See main `README.md`
