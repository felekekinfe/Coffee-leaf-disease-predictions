
import tensorflow as tf
from data_loader import get_data_generators
from model import build_cnn
import matplotlib.pyplot as plt

def train_model(train_dir='data/train', test_dir='data/test', model_path='models/my_coffee_leaf_model.h5'):
    # Load data
    train_generator, test_generator = get_data_generators(train_dir, test_dir)
    num_classes = len(train_generator.class_indices)
    
    # Build and train model
    model = build_cnn(num_classes=num_classes)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32,
        epochs=30,
        validation_data=test_generator,
        validation_steps=test_generator.samples // 32,
        callbacks=[early_stopping]
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Plot results
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
    
    # Save model
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()