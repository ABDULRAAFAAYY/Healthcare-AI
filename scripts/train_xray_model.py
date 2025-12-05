import os
import sys
import numpy as np
import tf_keras as keras
from tf_keras import layers
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_preprocessing import preprocess_image_data, augment_images
from backend.ml.model_loader import ModelLoader

def create_cnn_model(input_shape, num_classes):
    """
    Create CNN model for X-ray classification
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes
        
    Returns:
        keras.Model: Compiled CNN model
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_xray_model(image_dir, save_path, img_size=(224, 224), epochs=50, batch_size=32):
    """
    Train CNN model for X-ray classification
    
    Args:
        image_dir (str): Directory containing image data
        save_path (str): Path to save the trained model
        img_size (tuple): Target image size
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    """
    print("=" * 60)
    print("Training X-ray Classification Model")
    print("=" * 60)
    
    # Load and preprocess images
    images, labels, class_names = preprocess_image_data(image_dir, img_size)
    
    print(f"\nDataset Summary:")
    print(f"Total images: {len(images)}")
    print(f"Classes: {class_names}")
    print(f"Image shape: {images[0].shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Create data augmentation
    datagen = augment_images(X_train, y_train)
    
    # Create model
    print("\nCreating CNN model...")
    input_shape = X_train[0].shape
    num_classes = len(class_names)
    
    model = create_cnn_model(input_shape, num_classes)
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\nTraining model...")
    print("This may take a while depending on dataset size and hardware...")
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)
    
    # Plot training history
    plot_training_history(history, save_dir=os.path.dirname(save_path))
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {save_path}")
    
    return model, history

def plot_training_history(history, save_dir):
    """
    Plot and save training history
    
    Args:
        history: Keras training history object
        save_dir (str): Directory to save plots
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Save plot
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to: {plot_path}")
    
    plt.close()

def create_sample_image_structure(base_dir):
    """
    Create sample directory structure for X-ray images
    
    Args:
        base_dir (str): Base directory for image data
    """
    print("Creating sample image directory structure...")
    
    classes = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis', 'Lung_Cancer']
    
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    print(f"Created directories in: {base_dir}")
    print(f"Classes: {classes}")
    print("\nIMPORTANT: Add X-ray images to these directories before training!")
    print("Each directory should contain images for that specific class.")

if __name__ == "__main__":
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_dir = os.path.join(base_dir, 'data', 'xray_dataset')
    save_path = os.path.join(base_dir, 'backend', 'models', 'xray_model.h5')
    
    # Check if image directory exists and has data
    if not os.path.exists(image_dir) or len(os.listdir(image_dir)) == 0:
        print("X-ray dataset not found!")
        print("\n" + "=" * 60)
        print("Creating directory structure...")
        create_sample_image_structure(image_dir)
        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("1. Obtain a medical X-ray dataset (e.g., from Kaggle, NIH)")
        print("2. Organize images into the created class directories")
        print("3. Run this script again to train the model")
        print("=" * 60)
        sys.exit(0)
    
    # Train model
    try:
        train_xray_model(
            image_dir=image_dir,
            save_path=save_path,
            img_size=(224, 224),
            epochs=50,
            batch_size=32
        )
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
