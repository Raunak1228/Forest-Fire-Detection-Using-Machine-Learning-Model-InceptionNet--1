import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Step 1: Obtain a labeled dataset of thermal images

# Specify the path to your labeled training and test datasets
train_dataset_path = "C:/Users/HP/Desktop/6_SEM/Forest Fire Detection/Thermal Images Final Data/Train"
test_dataset_path = "C:/Users/HP/Desktop/6_SEM/Forest Fire Detection/Thermal Images Final Data/Test"

# Step 2: Prepare the dataset

# Initialize lists to store images and labels
train_images = []
train_labels = []
test_images = []
test_labels = []

# Iterate over the training dataset directory
for class_name in os.listdir(train_dataset_path):
    class_path = os.path.join(train_dataset_path, class_name)
    if os.path.isdir(class_path):
        # Iterate over the images in each class directory
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                # Load the image and convert it to a NumPy array
                image = load_img(image_path, target_size=(224, 224))
                image = img_to_array(image)
                # Normalize the image by dividing by 255
                image /= 255.0
                # Add the image and label to the training lists
                train_images.append(image)
                train_labels.append(class_name)

# Convert the training lists to NumPy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Iterate over the test dataset directory
for class_name in os.listdir(test_dataset_path):
    class_path = os.path.join(test_dataset_path, class_name)
    if os.path.isdir(class_path):
        # Iterate over the images in each class directory
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                # Load the image and convert it to a NumPy array
                image = load_img(image_path, target_size=(224, 224))
                image = img_to_array(image)
                # Normalize the image by dividing by 255
                image /= 255.0
                # Add the image and label to the test lists
                test_images.append(image)
                test_labels.append(class_name)

# Convert the test lists to NumPy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Step 3: Feature extraction or transfer learning

# Load the pre-trained VGG16 model without the top classification layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features from the pre-trained model using training images
train_features = base_model.predict(train_images)
test_features = base_model.predict(test_images)

# Step 4: Build a classifier

# Build the top classification layers
model = Sequential()
model.add(Flatten(input_shape=train_features.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Step 5: Train the model

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Step 6: Save the updated model to a new folder

updated_model_path = "C:/Users/HP/Desktop/6_SEM/Forest Fire Detection/model/Newly Trained Model/model.h5"

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(updated_model_path), exist_ok=True)

# Save the updated model
model.save(updated_model_path)
print("Updated model saved successfully.")
