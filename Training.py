import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up the training data generator
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'D:/Mini-Project/Final Data/Training',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

# Set up the testing data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        'D:/Mini-Project/Final Data/Test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

# Build the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=2, validation_data=test_generator)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size, verbose=2)
print('Test accuracy:', test_acc)
# Save the trained model to a specific file location
model.save('C:/Users/HP/Desktop/6_SEM/Forest Fire Detection/Forest Fire Detection_model.h5')
