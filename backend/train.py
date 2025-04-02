import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset paths
train_dir = "data/train"
val_dir = "data/val"

# Image parameters
IMG_SIZE = (224, 224)  # Original size
BATCH_SIZE = 32  # Original batch size
EPOCHS = 5  # Original epochs

# Apply Basic Data Preprocessing (NO Augmentation)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)
val_data = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)

# Load MobileNetV2 as Base Model (Fully Frozen)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False 

# Build the Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),  # No regularization
    Dropout(0.3),  # Lower dropout than fine-tuned models
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Save the Model
model.save("pneumonia_mobilenetv2.keras")

print("✅ Training complete. Model saved as pneumonia_mobilenetv2.keras")
