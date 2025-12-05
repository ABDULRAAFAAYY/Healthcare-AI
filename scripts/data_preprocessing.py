import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import os

def load_symptom_data(data_path):
    """
    Load and preprocess symptom dataset
    
    Args:
        data_path (str): Path to the CSV file
        
    Returns:
        tuple: (X, y, symptom_encoder, label_encoder)
    """
    print(f"Loading data from {data_path}...")
    
    # Load CSV
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    
    # Assuming CSV has 'symptoms' and 'disease' columns
    # symptoms should be comma-separated or already in list format
    
    # Clean and prepare symptoms
    if 'symptoms' in df.columns:
        # If symptoms are comma-separated strings
        if isinstance(df['symptoms'].iloc[0], str):
            df['symptoms'] = df['symptoms'].apply(
                lambda x: [s.strip().lower().replace(' ', '_') for s in x.split(',')]
            )
    
    # Extract features and labels
    X = df['symptoms'].tolist()
    y = df['disease'].tolist()
    
    # Encode symptoms using MultiLabelBinarizer
    symptom_encoder = MultiLabelBinarizer()
    X_encoded = symptom_encoder.fit_transform(X)
    
    # Encode disease labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Number of unique symptoms: {len(symptom_encoder.classes_)}")
    print(f"Number of disease classes: {len(label_encoder.classes_)}")
    print(f"Feature matrix shape: {X_encoded.shape}")
    
    return X_encoded, y_encoded, symptom_encoder, label_encoder

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Args:
        X: Feature matrix
        y: Labels
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def preprocess_image_data(image_dir, img_size=(224, 224)):
    """
    Preprocess medical images for training
    
    Args:
        image_dir (str): Directory containing images organized by class
        img_size (tuple): Target image size
        
    Returns:
        tuple: (images, labels, class_names)
    """
    from PIL import Image
    import os
    
    images = []
    labels = []
    class_names = []
    
    print(f"Loading images from {image_dir}...")
    
    # Iterate through subdirectories (each subdirectory is a class)
    for class_idx, class_name in enumerate(sorted(os.listdir(image_dir))):
        class_path = os.path.join(image_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        class_names.append(class_name)
        print(f"Processing class: {class_name}")
        
        # Load images from this class
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            
            try:
                # Load and preprocess image
                img = Image.open(img_path)
                
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize
                img = img.resize(img_size)
                
                # Convert to array and normalize
                img_array = np.array(img) / 255.0
                
                images.append(img_array)
                labels.append(class_idx)
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    print(f"Loaded {len(images)} images from {len(class_names)} classes")
    
    return np.array(images), np.array(labels), class_names

def augment_images(images, labels):
    """
    Apply data augmentation to images
    
    Args:
        images: Array of images
        labels: Array of labels
        
    Returns:
        tuple: (augmented_images, augmented_labels)
    """
    from tf_keras.preprocessing.image import ImageDataGenerator
    
    # Create data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    print("Data augmentation configured")
    return datagen

def balance_dataset(X, y):
    """
    Balance dataset by oversampling minority classes
    
    Args:
        X: Feature matrix
        y: Labels
        
    Returns:
        tuple: (X_balanced, y_balanced)
    """
    from sklearn.utils import resample
    
    unique_classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    
    X_balanced = []
    y_balanced = []
    
    for class_label in unique_classes:
        # Get samples for this class
        class_indices = np.where(y == class_label)[0]
        X_class = X[class_indices]
        y_class = y[class_indices]
        
        # Resample to match max count
        if len(X_class) < max_count:
            X_resampled, y_resampled = resample(
                X_class, y_class,
                n_samples=max_count,
                random_state=42
            )
        else:
            X_resampled, y_resampled = X_class, y_class
        
        X_balanced.append(X_resampled)
        y_balanced.append(y_resampled)
    
    X_balanced = np.vstack(X_balanced)
    y_balanced = np.concatenate(y_balanced)
    
    print(f"Balanced dataset: {len(X_balanced)} samples")
    
    return X_balanced, y_balanced

if __name__ == "__main__":
    print("Data Preprocessing Utilities")
    print("=" * 60)
    print("This module provides utilities for preprocessing medical data")
    print("Import these functions in your training scripts")
