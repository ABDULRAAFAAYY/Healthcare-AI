import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_preprocessing import load_symptom_data, split_data, balance_dataset
from backend.ml.model_loader import ModelLoader

def train_symptom_model(data_path, save_dir):
    """
    Train Random Forest model for symptom-based disease prediction
    
    Args:
        data_path (str): Path to symptom dataset CSV
        save_dir (str): Directory to save trained models
    """
    print("=" * 60)
    print("Training Symptom-Based Disease Prediction Model")
    print("=" * 60)
    
    # Load and preprocess data
    X, y, symptom_encoder, label_encoder = load_symptom_data(data_path)
    
    # Balance dataset
    print("\nBalancing dataset...")
    X_balanced, y_balanced = balance_dataset(X, y)
    
    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_data(X_balanced, y_balanced)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Define hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    print("\nPerforming hyperparameter tuning...")
    print("This may take several minutes...")
    
    # Create base model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy',
        verbose=1, n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_
    ))
    
    # Feature importance
    print("\nTop 10 Most Important Symptoms:")
    feature_importance = pd.DataFrame({
        'symptom': symptom_encoder.classes_,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10))
    
    # Save models
    print("\nSaving models...")
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'symptom_model.pkl')
    symptom_encoder_path = os.path.join(save_dir, 'symptom_encoder.pkl')
    label_encoder_path = os.path.join(save_dir, 'label_encoder.pkl')
    
    ModelLoader.save_pickle_model(best_model, model_path)
    ModelLoader.save_pickle_model(symptom_encoder, symptom_encoder_path)
    ModelLoader.save_pickle_model(label_encoder, label_encoder_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {model_path}")
    print(f"Symptom encoder saved to: {symptom_encoder_path}")
    print(f"Label encoder saved to: {label_encoder_path}")
    
    return best_model, symptom_encoder, label_encoder

def create_sample_dataset(save_path):
    """
    Create a sample symptom dataset for demonstration
    
    Args:
        save_path (str): Path to save the CSV file
    """
    print("Creating sample symptom dataset...")
    
    # Sample data
    data = {
        'symptoms': [
            'fever,cough,fatigue,difficulty_breathing',
            'fever,body_ache,fatigue,cough,headache',
            'runny_nose,sneezing,sore_throat,cough',
            'fever,cough,loss_of_taste,loss_of_smell',
            'sneezing,watery_eyes,runny_nose,congestion',
            'cough,chest_pain,difficulty_breathing,fatigue',
            'fever,cough,fatigue,body_ache',
            'runny_nose,sore_throat,sneezing,congestion',
            'fever,difficulty_breathing,cough,chest_pain',
            'watery_eyes,sneezing,runny_nose',
            'fever,body_ache,headache,fatigue,chills',
            'cough,wheezing,chest_pain,shortness_of_breath',
            'fever,cough,loss_of_smell,fatigue',
            'sore_throat,runny_nose,sneezing,cough',
            'fever,cough,difficulty_breathing,fatigue,headache',
            'congestion,sneezing,watery_eyes,runny_nose',
            'cough,chest_pain,fatigue,wheezing',
            'fever,body_ache,cough,fatigue,headache',
            'runny_nose,sneezing,sore_throat',
            'fever,loss_of_taste,cough,fatigue',
        ],
        'disease': [
            'COVID-19',
            'Influenza (Flu)',
            'Common Cold',
            'COVID-19',
            'Allergic Rhinitis',
            'Bronchitis',
            'Influenza (Flu)',
            'Common Cold',
            'COVID-19',
            'Allergic Rhinitis',
            'Influenza (Flu)',
            'Bronchitis',
            'COVID-19',
            'Common Cold',
            'COVID-19',
            'Allergic Rhinitis',
            'Bronchitis',
            'Influenza (Flu)',
            'Common Cold',
            'COVID-19',
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"Sample dataset created: {save_path}")
    print(f"Records: {len(df)}")
    print(f"Diseases: {df['disease'].unique().tolist()}")

if __name__ == "__main__":
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'symptoms_dataset.csv')
    save_dir = os.path.join(base_dir, 'backend', 'models')
    
    # Create sample dataset if it doesn't exist
    if not os.path.exists(data_path):
        print("Dataset not found. Creating sample dataset...")
        create_sample_dataset(data_path)
        print("\n" + "=" * 60)
        print("IMPORTANT: This is a minimal sample dataset!")
        print("For production use, you need a comprehensive medical dataset")
        print("with thousands of examples covering various diseases and symptoms.")
        print("=" * 60 + "\n")
    
    # Train model
    try:
        train_symptom_model(data_path, save_dir)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
