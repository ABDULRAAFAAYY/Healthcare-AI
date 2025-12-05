import pickle
import os
import numpy as np

class ModelLoader:
    """Utility class for loading ML models"""
    
    @staticmethod
    def load_pickle_model(model_path):
        """
        Load a pickled scikit-learn model
        
        Args:
            model_path (str): Path to the pickle file
            
        Returns:
            Loaded model or None if loading fails
        """
        try:
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found at {model_path}")
                return None
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            print(f"Successfully loaded model from {model_path}")
            return model
            
        except Exception as e:
            print(f"Error loading pickle model from {model_path}: {str(e)}")
            return None
    
    @staticmethod
    def load_keras_model(model_path):
        """
        Load a Keras/TensorFlow model
        
        Args:
            model_path (str): Path to the .h5 or SavedModel file
            
        Returns:
            Loaded model or None if loading fails
        """
        try:
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found at {model_path}")
                return None
            
            # Import keras only when needed
            try:
                import tf_keras as keras
            except ImportError:
                try:
                    from tensorflow import keras
                except ImportError:
                    print("Warning: Neither tf_keras nor tensorflow.keras available")
                    return None
            
            model = keras.models.load_model(model_path)
            print(f"Successfully loaded Keras model from {model_path}")
            return model
            
        except Exception as e:
            print(f"Error loading Keras model from {model_path}: {str(e)}")
            return None
    
    @staticmethod
    def save_pickle_model(model, save_path):
        """
        Save a model using pickle
        
        Args:
            model: Model to save
            save_path (str): Path where to save the model
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"Successfully saved model to {save_path}")
            
        except Exception as e:
            print(f"Error saving model to {save_path}: {str(e)}")
            raise
    
    @staticmethod
    def save_keras_model(model, save_path):
        """
        Save a Keras model
        
        Args:
            model: Keras model to save
            save_path (str): Path where to save the model
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path)
            print(f"Successfully saved Keras model to {save_path}")
            
        except Exception as e:
            print(f"Error saving Keras model to {save_path}: {str(e)}")
            raise
