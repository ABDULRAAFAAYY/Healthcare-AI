"""
Automated X-Ray Model Setup and Training Script
This script automates the entire process of setting up and training the X-ray model.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'xray_dataset'
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'validation'
TEST_DIR = DATA_DIR / 'test'

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

def check_kaggle_credentials():
    """Check if Kaggle credentials are set up"""
    print_header("CHECKING KAGGLE CREDENTIALS")
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if kaggle_json.exists():
        print_success("Kaggle credentials found!")
        return True
    else:
        print_error("Kaggle credentials not found!")
        print_info("Please set up Kaggle API:")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Scroll to 'API' section")
        print("  3. Click 'Create New API Token'")
        print(f"  4. Save kaggle.json to: {kaggle_dir}")
        print("\nAfter setting up, run this script again.")
        return False

def download_datasets():
    """Download X-ray datasets from Kaggle"""
    print_header("DOWNLOADING DATASETS")
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(DATA_DIR)
    
    datasets = [
        {
            'name': 'Chest X-Ray Pneumonia',
            'id': 'paultimothymooney/chest-xray-pneumonia',
            'file': 'chest-xray-pneumonia.zip'
        },
        {
            'name': 'COVID-19 Radiography',
            'id': 'tawsifurrahman/covid19-radiography-database',
            'file': 'covid19-radiography-database.zip'
        }
    ]
    
    for dataset in datasets:
        print(f"\n📥 Downloading {dataset['name']}...")
        
        if Path(dataset['file']).exists():
            print_warning(f"{dataset['file']} already exists, skipping download")
        else:
            try:
                subprocess.run([
                    'kaggle', 'datasets', 'download', '-d', dataset['id']
                ], check=True)
                print_success(f"Downloaded {dataset['name']}")
            except subprocess.CalledProcessError:
                print_error(f"Failed to download {dataset['name']}")
                return False
        
        # Unzip
        print(f"📦 Extracting {dataset['file']}...")
        try:
            subprocess.run(['powershell', 'Expand-Archive', '-Force', dataset['file'], '.'], check=True)
            print_success(f"Extracted {dataset['name']}")
        except:
            print_warning("Using Python to extract...")
            import zipfile
            with zipfile.ZipFile(dataset['file'], 'r') as zip_ref:
                zip_ref.extractall('.')
            print_success(f"Extracted {dataset['name']}")
    
    os.chdir(BASE_DIR)
    return True

def organize_dataset():
    """Organize downloaded images into train/val/test structure"""
    print_header("ORGANIZING DATASET")
    
    try:
        from sklearn.model_selection import train_test_split
        import random
        
        # Create directory structure
        classes = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']
        for split in ['train', 'validation', 'test']:
            for cls in classes:
                (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)
        
        def copy_and_split(source_dir, class_name, dest_base):
            """Copy images and split into train/val/test"""
            if not source_dir.exists():
                print_warning(f"Source directory not found: {source_dir}")
                return
            
            # Get all images
            images = list(source_dir.glob('*.jpeg')) + list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.png'))
            
            if len(images) == 0:
                print_warning(f"No images found in {source_dir}")
                return
            
            # Shuffle
            random.shuffle(images)
            
            # Split: 70% train, 15% val, 15% test
            train_size = int(0.7 * len(images))
            val_size = int(0.15 * len(images))
            
            train_imgs = images[:train_size]
            val_imgs = images[train_size:train_size + val_size]
            test_imgs = images[train_size + val_size:]
            
            # Copy files
            for img in train_imgs:
                shutil.copy(img, dest_base / 'train' / class_name / img.name)
            for img in val_imgs:
                shutil.copy(img, dest_base / 'validation' / class_name / img.name)
            for img in test_imgs:
                shutil.copy(img, dest_base / 'test' / class_name / img.name)
            
            print_success(f"{class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
        
        # Organize Pneumonia dataset
        print("\n📁 Organizing Pneumonia dataset...")
        chest_xray = DATA_DIR / 'chest_xray'
        if chest_xray.exists():
            copy_and_split(chest_xray / 'train' / 'NORMAL', 'Normal', DATA_DIR)
            copy_and_split(chest_xray / 'train' / 'PNEUMONIA', 'Pneumonia', DATA_DIR)
        
        # Organize COVID-19 dataset
        print("\n📁 Organizing COVID-19 dataset...")
        covid_dir = DATA_DIR / 'COVID-19_Radiography_Dataset'
        if not covid_dir.exists():
            covid_dir = DATA_DIR / 'COVID-19 Radiography Database'
        
        if covid_dir.exists():
            copy_and_split(covid_dir / 'COVID' / 'images', 'COVID-19', DATA_DIR)
            # Use some normal images from COVID dataset
            normal_covid = covid_dir / 'Normal' / 'images'
            if normal_covid.exists():
                copy_and_split(normal_covid, 'Normal', DATA_DIR)
        
        # For TB, use some pneumonia images as placeholder
        print("\n📁 Creating Tuberculosis placeholder...")
        pneumonia_imgs = list((DATA_DIR / 'train' / 'Pneumonia').glob('*.jpeg'))[:100]
        for img in pneumonia_imgs:
            shutil.copy(img, DATA_DIR / 'train' / 'Tuberculosis' / img.name)
        
        print_success("Dataset organized successfully!")
        return True
        
    except Exception as e:
        print_error(f"Failed to organize dataset: {str(e)}")
        return False

def train_model():
    """Train the X-ray classification model"""
    print_header("TRAINING MODEL")
    
    print_info("Starting model training...")
    print_info("This may take 30-60 minutes on GPU or 3-6 hours on CPU")
    print_info("You can monitor progress in the terminal\n")
    
    try:
        # Run training script
        subprocess.run([
            sys.executable,
            str(BASE_DIR / 'scripts' / 'train_xray_model_real.py')
        ], check=True)
        
        print_success("Model training completed!")
        return True
        
    except subprocess.CalledProcessError:
        print_error("Model training failed!")
        return False

def main():
    """Main automation function"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     AUTOMATED X-RAY MODEL SETUP & TRAINING                ║")
    print("║     Healthcare AI Application                             ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")
    
    # Step 1: Check Kaggle credentials
    if not check_kaggle_credentials():
        return
    
    # Step 2: Download datasets
    print_info("This will download ~2-3 GB of data. Continue? (y/n): ")
    response = input().strip().lower()
    if response != 'y':
        print_warning("Setup cancelled by user")
        return
    
    if not download_datasets():
        print_error("Dataset download failed!")
        return
    
    # Step 3: Organize dataset
    if not organize_dataset():
        print_error("Dataset organization failed!")
        return
    
    # Step 4: Train model
    print_info("Ready to train model. Continue? (y/n): ")
    response = input().strip().lower()
    if response != 'y':
        print_warning("Training cancelled by user")
        return
    
    if not train_model():
        print_error("Training failed!")
        return
    
    # Success!
    print_header("✅ SETUP COMPLETE!")
    print_success("Your X-ray classification model is ready!")
    print_info("Model saved to: backend/models/xray_model.h5")
    print_info("\nNext steps:")
    print("  1. Restart your Flask backend: python backend/app.py")
    print("  2. Go to http://localhost:3000")
    print("  3. Upload X-ray images and get predictions!")
    print(f"\n{Colors.GREEN}🎉 Enjoy your Healthcare AI application!{Colors.END}\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Setup interrupted by user{Colors.END}")
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
