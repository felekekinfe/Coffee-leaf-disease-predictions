
"""Module to load a trained model and predict coffee leaf diseases from images."""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

def predict_image(model_path='models/my_coffee_leaf_model.h5', img_path='test_image.jpg'):
    """
    Load a trained model and predict the class of a single image.
    
    Args:
        model_path (str): Path to the trained model file, default is 'models/my_coffee_leaf_model.h5'.
        img_path (str): Path to the image to predict, default is 'test_image.jpg'.
    
    Raises:
        FileNotFoundError: If model or image file is missing.
        ValueError: If prediction fails.
    """
    # Validate file existence
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    try:
        # Load the model
        model = load_model(model_path)
        print("Model loaded successfully!")
        
        # Load and preprocess the image
        img_size = (224, 224)
        img = image.load_img(img_path, target_size=img_size)
        plt.imshow(img)
        plt.title("Test Image")
        plt.axis('off')
        plt.show()
        
        img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(img_array)
        class_names = ['Healthy', 'Miner', 'Phoma', 'Rust']  # Adjust based on your dataset
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Display results
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e))
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    """Test prediction with default paths."""
    try:
        predict_image(img_path='path_to_your_image.jpg')  # Replace with your image path
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")