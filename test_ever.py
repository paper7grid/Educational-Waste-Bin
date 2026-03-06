# test_every.py
# Test camera + AI model with TensorFlow

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time

print("="*50)
print("TESTING CAMERA + AI MODEL")
print("="*50)

# === TEST 1: Load AI Model ===
print("\n[TEST 1] Loading AI model...")
try:
    interpreter = tf.lite.Interpreter(model_path='model_unquant.tflite')
    interpreter.allocate_tensors()
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ ERROR: {e}")
    exit()

# === TEST 2: Load Labels ===
print("\n[TEST 2] Loading labels...")
try:
    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"✓ Labels loaded: {labels}")
except Exception as e:
    print(f"✗ ERROR: {e}")
    exit()

# === TEST 3: Start Camera ===
print("\n[TEST 3] Starting camera...")
from picamera2 import Picamera2

camera = Picamera2()
camera_config = camera.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
camera.configure(camera_config)
camera.set_controls({
    "AwbEnable": True,           # Enable auto white balance
    "AwbMode": 0,                # Auto mode (0 = auto, 1 = incandescent, etc.)
    "AeEnable": True,            # Enable auto exposure
    "Brightness": 0.0,           # Neutral brightness
    "Contrast": 1.0              # Normal contrast
})
camera.start()
time.sleep(2)
print("✓ Camera started!")


print("\n" + "="*50)
print("Starting live camera + AI test...")
print("Press 'q' in the video window to quit")
print("="*50 + "\n")

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Interpreter done")

# Main loop
frame_count = 0
prediction = "Loading..."
confidence = 0

try:
    while True:
        frame = camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        ret = True  # picamera2 doesn't return ret
       
        if not ret:
            break
       
        frame_count += 1
       
        # Classify every 30 frames
        if frame_count % 30 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            small_image = pil_image.resize((224, 224))
           
            img_array = np.array(small_image, dtype=np.float32)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
           
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]
           
            best_index = np.argmax(predictions)
            confidence = predictions[best_index] * 100
            prediction = labels[best_index]
           
            print(f"🎯 {prediction} ({confidence:.1f}%)")
       
        # Draw on screen
        cv2.rectangle(frame, (5, 5), (635, 90), (0, 0, 0), -1)
        text = f"{prediction} - {confidence:.0f}%"
        cv2.putText(frame, text, (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
       
        cv2.imshow('Camera + AI Test', frame)
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopped")

finally:
    camera.stop()
    cv2.destroyAllWindows()
    print("Done!")
