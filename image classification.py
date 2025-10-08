# ==========================================
# IMPORTS
# ==========================================
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Updated import
from tensorflow.keras import layers, models
# ==========================================
# CONFIGURATION
# ==========================================
img_height = 150
img_width = 150
batch_size = 32
epochs = 5  # ✅ You can change this for better training
class_labels = ['degradable', 'non degradable']

# ==========================================
# DATA PREPARATION
# ==========================================
train_dir = "waste_dataset/train"
val_dir = "waste_dataset/val"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# ==========================================
# MODEL CREATION
# ==========================================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================
# TRAINING PHASE
# ==========================================
print("\n🚀 Starting model training...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

# Plot training graph
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# ==========================================
# SAVE THE MODEL
# ==========================================
save_path = r"C:\dhivya folder\image\waste_classification.h5"
model.save(save_path)
print(f"\n✅ Model saved successfully at: {save_path}")

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_image(img_path, model, class_labels):
    img = load_img(img_path, target_size=(img_height, img_width))  # Use load_img
    img_array = img_to_array(img)  # Use img_to_array
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = int(prediction[0][0] > 0.5)
    label = class_labels[predicted_class]

    print(f"🧾 Predicted: {label}")
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {label}")
    plt.show()

# ==========================================
# CONTINUOUS IMAGE PREDICTION
# ==========================================
while True:
    img_path = input("\nEnter the path to the image you want to classify (or type 'exit' to quit): ")
    if img_path.strip().lower() == 'exit' or img_path.strip() == '':
        print("👋 Exiting.")
        break
    elif not os.path.exists(img_path):
        print("⚠️ Invalid path. Please try again.")
        continue
    predict_image(img_path, model, class_labels)
