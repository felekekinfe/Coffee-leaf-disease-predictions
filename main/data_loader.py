

"""Module to create data generators for training and testing image datasets."""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def get_data_generators(train_dir, test_dir, img_size=(224, 224), batch_size=32):
    """
    Create data generators for training and testing image datasets with preprocessing.
    
    Args:
        train_dir (str): Path to the training dataset directory.
        test_dir (str): Path to the test dataset directory.
        img_size (tuple): Target image size (width, height), default is (224, 224).
        batch_size (int): Number of images per batch, default is 32.
    
    Returns:
        tuple: (train_generator, test_generator) for feeding into a model.
    
    Raises:
        ValueError: If directories don’t exist or are empty.
    """
    
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise ValueError(f"Directories not found: train_dir={train_dir}, test_dir={test_dir}")
    if not os.listdir(train_dir) or not os.listdir(test_dir):
        raise ValueError("One or both directories are empty")

    try:
        # Training data generator with augmentation for robustness
        train_datagen = ImageDataGenerator(
            rescale=1./255,          # Normalize pixel values to [0, 1]
            rotation_range=20,       # Random rotation up to 20 degrees
            width_shift_range=0.2,   # Random horizontal shift
            height_shift_range=0.2,  # Random vertical shift
            shear_range=0.2,         # Random shear transformation
            zoom_range=0.2,          # Random zoom
            horizontal_flip=True,    # Random horizontal flip
            fill_mode='nearest'      # Fill new pixels with nearest values
        )

        # Test data generator with only rescaling
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Load training data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'  # Multi-class labels as one-hot vectors
        )

        # Load test data
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False             # Maintain order for evaluatio
        )

        return train_generator, test_generator

    except Exception as e:
        raise ValueError(f"Error creating generators: {str(e)}")

if __name__ == "__main__":
    """Test the data loader with example paths."""
    try:
        # Example paths (replace with your actual dataset paths)
        train_dir = 'data/train'
        test_dir = 'data/test'
        
        # Create generators
        train_gen, test_gen = get_data_generators(train_dir, test_dir)
        
        # Display results
        print(f"Number of classes: {len(train_gen.class_indices)}")
        print(f"Class labels: {train_gen.class_indices}")
        
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")