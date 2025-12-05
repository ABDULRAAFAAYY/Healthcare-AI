# X-Ray Dataset Setup Guide

## Quick Start

This guide will help you set up a real medical X-ray dataset for training the classification model.

## Dataset Sources

### Option 1: Kaggle Datasets (Recommended)

#### 1. Install Kaggle API
```bash
pip install kaggle
```

#### 2. Set up Kaggle API credentials
- Go to https://www.kaggle.com/account
- Click "Create New API Token"
- Save `kaggle.json` to `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

#### 3. Download Datasets

**Chest X-Ray Pneumonia Dataset:**
```bash
cd data/xray_dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip
```

**COVID-19 Radiography Database:**
```bash
kaggle datasets download -d tawsifurrahman/covid19-radiography-database
unzip covid19-radiography-database.zip
```

**Tuberculosis X-Ray Dataset:**
```bash
kaggle datasets download -d usmanshams/tbx-11
unzip tbx-11.zip
```

### Option 2: Manual Download

1. **Chest X-Ray Images (Pneumonia)**
   - URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
   - Download and extract to `data/xray_dataset/`

2. **COVID-19 Radiography Database**
   - URL: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
   - Download and extract to `data/xray_dataset/`

3. **Tuberculosis Dataset**
   - URL: https://www.kaggle.com/datasets/usmanshams/tbx-11
   - Download and extract to `data/xray_dataset/`

## Directory Structure

After downloading, organize your data like this:

```
data/xray_dataset/
├── train/
│   ├── Normal/           (500+ images)
│   ├── Pneumonia/        (500+ images)
│   ├── COVID-19/         (500+ images)
│   ├── Tuberculosis/     (500+ images)
│   └── Lung_Cancer/      (if available)
├── validation/
│   ├── Normal/           (100+ images)
│   ├── Pneumonia/        (100+ images)
│   ├── COVID-19/         (100+ images)
│   ├── Tuberculosis/     (100+ images)
│   └── Lung_Cancer/      (if available)
└── test/
    ├── Normal/           (100+ images)
    ├── Pneumonia/        (100+ images)
    ├── COVID-19/         (100+ images)
    ├── Tuberculosis/     (100+ images)
    └── Lung_Cancer/      (if available)
```

## Data Organization Script

Use this Python script to organize downloaded data:

```python
import os
import shutil
from sklearn.model_selection import train_test_split

def organize_dataset(source_dir, dest_dir, class_name):
    """Organize images into train/val/test splits"""
    
    # Get all image files
    images = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Split: 70% train, 15% validation, 15% test
    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
    
    # Create directories
    for split in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(dest_dir, split, class_name), exist_ok=True)
    
    # Copy files
    for img in train_imgs:
        shutil.copy(
            os.path.join(source_dir, img),
            os.path.join(dest_dir, 'train', class_name, img)
        )
    
    for img in val_imgs:
        shutil.copy(
            os.path.join(source_dir, img),
            os.path.join(dest_dir, 'validation', class_name, img)
        )
    
    for img in test_imgs:
        shutil.copy(
            os.path.join(source_dir, img),
            os.path.join(dest_dir, 'test', class_name, img)
        )
    
    print(f"{class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

# Example usage:
# organize_dataset('downloaded/NORMAL', 'data/xray_dataset', 'Normal')
# organize_dataset('downloaded/PNEUMONIA', 'data/xray_dataset', 'Pneumonia')
# organize_dataset('downloaded/COVID', 'data/xray_dataset', 'COVID-19')
# organize_dataset('downloaded/TB', 'data/xray_dataset', 'Tuberculosis')
```

## Minimum Requirements

- **Minimum images per class**: 500 (for decent accuracy)
- **Recommended images per class**: 1000+
- **Image format**: JPEG or PNG
- **Image size**: Any (will be resized to 224x224)

## Data Quality Tips

1. **Remove duplicates**: Check for duplicate images
2. **Verify labels**: Ensure images are in correct folders
3. **Check image quality**: Remove corrupted or low-quality images
4. **Balance classes**: Try to have similar number of images per class

## Next Steps

After organizing your dataset:

1. Run the training script:
   ```bash
   python scripts/train_xray_model_real.py
   ```

2. The trained model will be saved to:
   ```
   backend/models/xray_model.h5
   ```

3. Test the model through the web interface!

## Troubleshooting

**Issue**: Not enough images for a class
- **Solution**: Use data augmentation (already included in training script)

**Issue**: Imbalanced dataset
- **Solution**: Use class weights (included in training script)

**Issue**: Low accuracy
- **Solution**: 
  - Get more data
  - Train for more epochs
  - Adjust learning rate
  - Try different base models

## Important Note

⚠️ **Medical Disclaimer**: This model is for **educational purposes only**. It should never be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice.
