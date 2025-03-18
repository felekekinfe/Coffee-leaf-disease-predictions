
"""Module to define a custom CNN for coffee leaf disease classification."""

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.models import Model

def build_cnn(input_shape=(224, 224, 3), num_classes=4):
    """
    Build a custom CNN for multiclass image classification.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels), default is (224, 224, 3).
        num_classes (int): Number of output classes, default is 4.
    
    Returns:
        Model: Compiled Keras model ready for training.
    
    Raises:
        ValueError: If num_classes is less than 2 or input_shape is invalid.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if len(input_shape) != 3 or input_shape[2] != 3:
        raise ValueError("input_shape must be (height, width, 3)")

    try:
        # Define the input layer
        inputs = Input(shape=input_shape)
        
        # Convolutional layers to extract features
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)  # 32 filters, 3x3 kernel
        x = MaxPooling2D((2, 2))(x)  # Reduce size to 112x112
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)  # Reduce to 56x56
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)  # Reduce to 28x28
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)  # Reduce to 14x14
        
        # Flatten and classify
        x = Flatten()(x)  # Convert to 1D vector
        x = Dense(512, activation='relu')(x)  # Fully connected layer
        x = Dropout(0.5)(x)  # 50% dropout to prevent overfitting
        outputs = Dense(num_classes, activation='softmax')(x)  # Output probabilities
        
        # Build and compile the model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        
        return model
    
    except Exception as e:
        raise ValueError(f"Error building model: {str(e)}")

if __name__ == "__main__":
    """Test the model creation."""
    try:
        model = build_cnn()
        model.summary()
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")