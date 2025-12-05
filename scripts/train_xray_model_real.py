"""
Real X-Ray Classification Model Training Script
Uses Transfer Learning with ResNet50 for chest X-ray classification

Classes: Normal, Pneumonia, COVID-19, Tuberculosis, Lung Cancer
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Check if TensorFlow/Keras is available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    KERAS_AVAILABLE = True
except ImportError:
    print("ERROR: TensorFlow/Keras not installed!")
    print("Install with: pip install tensorflow")
    KERAS_AVAILABLE = False
    exit(1)

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10  # Training with frozen base
EPOCHS_PHASE2 = 20  # Fine-tuning
LEARNING_RATE_PHASE1 = 0.001
LEARNING_RATE_PHASE2 = 0.0001

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'xray_dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'backend', 'models')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Class names
CLASS_NAMES = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis', 'Lung_Cancer']

def check_dataset():
    """Check if dataset exists and is properly organized"""
    print("=" * 60)
    print("CHECKING DATASET")
    print("=" * 60)
    
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Training directory not found: {TRAIN_DIR}")
        print("\n📋 Please set up your dataset first!")
        print("See: data/xray_dataset/SETUP_GUIDE.md")
        return False
    
    print(f"✅ Dataset directory found: {DATA_DIR}\n")
    
    # Count images per class
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir):
            print(f"⚠️  {split} directory not found")
            continue
            
        print(f"{split.upper()}:")
        total = 0
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  {class_name:20s}: {count:4d} images")
                total += count
            else:
                print(f"  {class_name:20s}:    0 images (directory missing)")
        print(f"  {'TOTAL':20s}: {total:4d} images\n")
    
    return True

def create_data_generators():
    """Create data generators with augmentation"""
    print("=" * 60)
    print("CREATING DATA GENERATORS")
    print("=" * 60)
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation/Test data (only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Test generator (if test directory exists)
    test_generator = None
    if os.path.exists(TEST_DIR):
        test_generator = val_test_datagen.flow_from_directory(
            TEST_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
    
    print(f"\n✅ Data generators created")
    print(f"   Classes: {train_generator.class_indices}")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    if test_generator:
        print(f"   Test samples: {test_generator.samples}")
    
    return train_generator, val_generator, test_generator

def create_model(num_classes):
    """Create model using ResNet50 transfer learning"""
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)
    
    # Load pre-trained ResNet50 (without top layers)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Create custom top layers
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    print(f"✅ Model created with ResNet50 base")
    print(f"   Input shape: {IMG_SIZE + (3,)}")
    print(f"   Output classes: {num_classes}")
    print(f"   Base model frozen: Yes")
    
    return model, base_model

def train_phase1(model, train_gen, val_gen):
    """Phase 1: Train with frozen base model"""
    print("\n" + "=" * 60)
    print("PHASE 1: TRAINING WITH FROZEN BASE")
    print("=" * 60)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'xray_model_phase1.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    history1 = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE1,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n✅ Phase 1 complete")
    return history1

def train_phase2(model, base_model, train_gen, val_gen):
    """Phase 2: Fine-tune with unfrozen layers"""
    print("\n" + "=" * 60)
    print("PHASE 2: FINE-TUNING")
    print("=" * 60)
    
    # Unfreeze last layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # Freeze all but last 30 layers
        layer.trainable = False
    
    print(f"   Unfrozen layers: {sum([1 for l in base_model.layers if l.trainable])}")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_PHASE2),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'xray_model.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-8,
            verbose=1
        )
    ]
    
    # Train
    history2 = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE2,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n✅ Phase 2 complete")
    return history2

def evaluate_model(model, test_gen):
    """Evaluate model on test set"""
    if test_gen is None:
        print("\n⚠️  No test set available for evaluation")
        return
    
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    
    print(f"\n📊 Test Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_accuracy*100:.2f}%")

def plot_training_history(history1, history2):
    """Plot training history"""
    print("\n" + "=" * 60)
    print("GENERATING TRAINING PLOTS")
    print("=" * 60)
    
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(x=EPOCHS_PHASE1, color='r', linestyle='--', label='Fine-tuning starts')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(x=EPOCHS_PHASE1, color='r', linestyle='--', label='Fine-tuning starts')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(MODEL_DIR, 'training_history.png')
    plt.savefig(plot_path)
    print(f"✅ Training plot saved: {plot_path}")
    
    plt.show()

def main():
    """Main training function"""
    print("\n" + "=" * 60)
    print("X-RAY CLASSIFICATION MODEL TRAINING")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check dataset
    if not check_dataset():
        return
    
    # Create output directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators()
    num_classes = len(train_gen.class_indices)
    
    # Create model
    model, base_model = create_model(num_classes)
    
    # Phase 1: Train with frozen base
    history1 = train_phase1(model, train_gen, val_gen)
    
    # Phase 2: Fine-tune
    history2 = train_phase2(model, base_model, train_gen, val_gen)
    
    # Evaluate
    evaluate_model(model, test_gen)
    
    # Plot results
    plot_training_history(history1, history2)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model saved to: {os.path.join(MODEL_DIR, 'xray_model.h5')}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎉 You can now use the model in your web application!")

if __name__ == '__main__':
    main()
