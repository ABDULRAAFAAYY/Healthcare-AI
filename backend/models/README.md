# ML Models Directory

This directory contains the trained machine learning models for the Healthcare AI application.

## Required Model Files

### Symptom-Based Prediction Models
- **symptom_model.pkl**: Random Forest classifier trained on symptom-disease dataset
- **symptom_encoder.pkl**: MultiLabelBinarizer for encoding symptoms
- **label_encoder.pkl**: LabelEncoder for disease labels

### Image-Based Prediction Model
- **xray_model.h5**: Convolutional Neural Network for X-ray image classification

## Training the Models

To train the models, use the training scripts in the `scripts/` directory:

```bash
# Train symptom-based model
python scripts/train_symptom_model.py

# Train X-ray classification model
python scripts/train_xray_model.py
```

## Model Details

### Symptom Model
- **Algorithm**: Random Forest Classifier
- **Input**: Binary vector of symptoms (presence/absence)
- **Output**: Disease prediction with probability scores
- **Features**: 30+ common symptoms
- **Classes**: 5+ disease categories

### X-ray Model
- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: 224x224 RGB images
- **Output**: Multi-class classification (Normal, Pneumonia, COVID-19, TB, Lung Cancer)
- **Preprocessing**: Normalization, resizing, augmentation

## Important Notes

⚠️ **Medical Disclaimer**: These models are for educational and demonstration purposes only. They should NOT be used for actual medical diagnosis without proper validation and regulatory approval.

⚠️ **Data Requirements**: Training requires substantial medical datasets that are not included in this repository due to size and privacy concerns.

⚠️ **Performance**: Model accuracy depends heavily on the quality and quantity of training data.

## Fallback Behavior

If model files are not present, the application will use rule-based demo predictions to allow testing of the UI and API without trained models.
