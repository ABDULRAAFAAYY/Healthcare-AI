# Healthcare AI - Data Directory

This directory contains datasets for training the Healthcare AI models.

## Directory Structure

```
data/
├── symptoms_dataset.csv       # Symptom-disease mapping dataset
└── xray_dataset/             # X-ray images organized by condition
    ├── Normal/
    ├── Pneumonia/
    ├── COVID-19/
    ├── Tuberculosis/
    └── Lung_Cancer/
```

## Symptoms Dataset

**File**: `symptoms_dataset.csv`

### Format
- **symptoms**: Comma-separated list of symptoms
- **disease**: Disease label

### Example
```csv
symptoms,disease
"fever,cough,fatigue",COVID-19
"runny_nose,sneezing",Common Cold
```

### Current Dataset
- 40 sample records
- 5 disease categories:
  - COVID-19
  - Influenza (Flu)
  - Common Cold
  - Allergic Rhinitis
  - Bronchitis

### Expanding the Dataset

For production use, you should:

1. **Obtain Medical Datasets**: Use reputable sources like:
   - Kaggle medical datasets
   - UCI Machine Learning Repository
   - Medical research databases

2. **Increase Sample Size**: Aim for:
   - Minimum 1000+ records
   - Balanced distribution across diseases
   - Diverse symptom combinations

3. **Add More Diseases**: Include additional conditions relevant to your use case

4. **Validate Data**: Ensure medical accuracy with healthcare professionals

## X-Ray Dataset

**Directory**: `xray_dataset/`

### Structure
Organize images into subdirectories by condition:

```
xray_dataset/
├── Normal/           # Normal chest X-rays
├── Pneumonia/        # Pneumonia cases
├── COVID-19/         # COVID-19 cases
├── Tuberculosis/     # TB cases
└── Lung_Cancer/      # Lung cancer cases
```

### Image Requirements

- **Format**: JPG, PNG, or JPEG
- **Size**: Minimum 224x224 pixels (will be resized)
- **Quality**: Clear, high-resolution medical images
- **Quantity**: Minimum 100+ images per class for basic training

### Recommended Datasets

1. **NIH Chest X-Ray Dataset**
   - 100,000+ chest X-rays
   - Multiple conditions
   - Publicly available

2. **COVID-19 Radiography Database**
   - COVID-19, Normal, Viral Pneumonia
   - Available on Kaggle

3. **RSNA Pneumonia Detection Challenge**
   - Large pneumonia dataset
   - Available on Kaggle

### Data Augmentation

The training script automatically applies:
- Rotation (±20 degrees)
- Width/height shifts (±20%)
- Horizontal flips
- Zoom (±20%)

## Privacy and Ethics

⚠️ **Important Considerations**:

1. **Patient Privacy**: Ensure all medical images are de-identified
2. **Data Rights**: Only use datasets you have permission to use
3. **Ethical Use**: Follow medical ethics guidelines
4. **Regulatory Compliance**: Comply with HIPAA, GDPR, or local regulations

## Training Models

Once you have datasets:

```bash
# Train symptom model
python scripts/train_symptom_model.py

# Train X-ray model
python scripts/train_xray_model.py
```

## Demo Mode

The application works without trained models using intelligent fallback logic. This allows you to:
- Test the UI and workflow
- Demonstrate the concept
- Develop and debug without large datasets

## Data Quality Tips

1. **Symptom Data**:
   - Use consistent symptom naming
   - Include symptom variations
   - Balance disease distribution
   - Validate medical accuracy

2. **Image Data**:
   - Use high-quality medical images
   - Ensure proper labeling
   - Include diverse cases
   - Verify with radiologists

## Contributing Data

If you have access to medical datasets:
1. Ensure proper permissions
2. De-identify all patient information
3. Validate data quality
4. Document data sources
5. Follow ethical guidelines

---

**Note**: The sample dataset provided is for demonstration purposes only and should not be used for actual medical applications.
