import cv2
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="pruned_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open camera
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()

    # Preprocess image
    img = cv2.resize(frame, (224, 224))
    img = img[np.newaxis, :, :, :]  # Add batch dimension

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])

    # Process prediction

    # Display image with overlay
    cv2.imshow('Segmentation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
