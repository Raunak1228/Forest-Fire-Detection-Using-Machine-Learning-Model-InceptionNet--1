import cv2
import tensorflow as tf
from twilio.rest import Client

# Load the trained model
model = tf.keras.models.load_model("D:/Mini-Project/Forest Fire Detection/model/my_running_model.h5")

# Twilio account credentials
account_sid = 'AC8a43e1b89274f9d62be3ed72a01812ab'
auth_token = '486cd4d26f1aa294f41000a6786f00a6'
twilio_phone_number = '+12707516892'
recipient_phone_number = '+919834629512'

# Fire detection parameters
fire_threshold = 0.5
fire_detected = False

# Twilio client
client = Client(account_sid, auth_token)

url = 'http://172.16.161.190:4747/video'
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (resize, normalize, etc.)
    frame = cv2.resize(frame, (224, 224))  # Resize the frame to match the model's input shape
    frame = frame / 255.0  # Normalize the pixel values between 0 and 1

    # Expand dimensions to create a batch of size 1
    frame = tf.expand_dims(frame, axis=0)

    # Make prediction using the model
    prediction = model.predict(frame)

    # Assuming a single output for fire detection
    fire_probability = prediction[0][0]

    # Decide whether it's a fire or not based on the probability threshold
    if fire_probability > fire_threshold:
        if not fire_detected:
            fire_detected = True
            message = client.messages.create(
                body='A forest fire has been detected in the ece dept. The fire is currently at huge and is spreading rapidly. Please evacuate the area immediately and call 108 if you see the fire.',
                from_=twilio_phone_number,
                to=recipient_phone_number
            )
            print('Fire detected! SMS sent.')
    else:
        fire_detected = False
        message = client.messages.create(
            body='A forest fire has not been detected in the ECE dept. The fire danger is currently low. Please continue to be aware of your surroundings and report any suspicious activity to 108..',
            from_=twilio_phone_number,
            to=recipient_phone_number
        )
        print('No fire detected. SMS sent.')

    # Convert the frame to a NumPy array for OpenCV operations
    frame = frame.numpy().squeeze()

    # Convert the color space from RGB to BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the frame with the fire probability
    label = f'Fire Probability: {fire_probability:.2f}'
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Fire Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
