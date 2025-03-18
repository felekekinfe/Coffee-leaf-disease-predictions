
"""Module to train the CNN model for coffee leaf disease classification."""

import tensorflow as tf
from data_loader import get_data_generators
from model import build_cnn
import matplotlib.pyplot as plt
import os

def train_model(train_dir='data/train', test_dir='data/test', model_path='models/my_coffee_leaf_model.h5'):
    """
    Train the CNN model and save it to disk.
    
    Args:
        train_dir (str): Path to training data, default is 'data/train'.
        test_dir (str): Path to test data, default is 'data/test'.
        model_path (str): Path to save the trained model, default is 'models/my_coffee_leaf_model.h5'.
    
    Raises:
        ValueError: If data loading or training fails.
        OSError: If model cannot be saved.
    """
    try:
        # Load data
        train_generator, test_generator = get_data_generators(train_dir, test_dir)
        num_classes = len(train_generator.class_indices)
        
        # Build model
        model = build_cnn(num_classes=num_classes)
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // 32,
            epochs=30,
            validation_data=test_generator,
            validation_steps=test_generator.samples // 32,
            callbacks=[early_stopping]
        )
        
        # Evaluate on test data
        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Test Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        # Save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
    except ValueError as e:
        raise ValueError(f"Training failed: {str(e)}")
    except OSError as e:
        raise OSError(f"Could not save model to {model_path}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error during training: {str(e)}")

if __name__ == "__main__":
    """Run training with default paths."""
    try:
        train_model()
    except (ValueError, OSError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")