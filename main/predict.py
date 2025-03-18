
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def predict_image(model_path='models/my_coffee_leaf_model.h5', img_path='test_image.jpg'):
    # Load model
    model = load_model(model_path)
    print("Model loaded successfully!")
    
    # Load and prepare image
    img_size = (224, 224)
    img = image.load_img(img_path, target_size=img_size)
    plt.imshow(img)
    plt.title("Test Image")
    plt.axis('off')
    plt.show()
    
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)
    class_names = ['Healthy', 'Miner', 'Phoma', 'Rust']  # Update based on your dataset
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    predict_image(img_path='')  