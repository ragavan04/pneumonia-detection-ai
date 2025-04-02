import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define test dataset path
test_dir = "data/test"

# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data preprocessing (rescale pixel values)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load test dataset
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)

# Load the original model
original_model = tf.keras.models.load_model("pneumonia_mobilenetv2.h5")
original_loss, original_accuracy = original_model.evaluate(test_data)

# Print comparison results
print("\n🔍 **Model Comparison Results:**")
print(f"Original Model - Accuracy: {original_accuracy:.4f}, Loss: {original_loss:.4f}")
