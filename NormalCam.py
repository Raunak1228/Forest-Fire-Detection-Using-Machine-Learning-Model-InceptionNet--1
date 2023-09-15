import cv2
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('"C:/Users/HP/Desktop/6_SEM/Forest Fire Detection/model/Forest Fire Detection_model.h5"')

# Set up the camera
camera = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    # Preprocess the frame (resize, normalize, etc.)
    frame = cv2.resize(frame, (224, 224))  # Resize the frame to match the model's input shape
    frame = frame / 255.0  # Normalize the pixel values between 0 and 1

    # Expand dimensions to create a batch of size 1
    frame = tf.expand_dims(frame, axis=0)

    # Make prediction using the model
    prediction = model.predict(frame)

    # Assuming a single output for fire detection
    fire_probability = prediction[0][0]

    # Decide whether it's a forest fire or not based on the probability threshold
    if fire_probability > 0.5:  # Adjust the threshold as per your model's performance
        label = 'Fire'
    else:
        label = 'Non-Fire'

    # Display the frame with the prediction label
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Forest Fire Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
