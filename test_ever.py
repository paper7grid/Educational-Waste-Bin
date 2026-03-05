#test_everything.py
# Test camera + AI model together

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import time

print("="*50)
print("TESTING CAMERA + AI MODEL")
print("="*50)

# === TEST 1: Load AI Model ===
print("\n[TEST 1] Loading AI model...")
try:
    interpreter = tflite.Interpreter(model_path='model_unquant.tflite')
    interpreter.allocate_tensors()
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ ERROR loading model: {e}")
    print("Make sure model.tflite is in this folder!")
    exit()

# === TEST 2: Load Labels ===
print("\n[TEST 2] Loading labels...")
try:
    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"✓ Labels loaded: {labels}")
except Exception as e:
    print(f"✗ ERROR loading labels: {e}")
    print("Make sure labels.txt is in this folder!")
    exit()

# === TEST 3: Start Camera ===
print("\n[TEST 3] Starting camera...")
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("✗ ERROR: Cannot open camera!")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(2)
print("✓ Camera started!")

# === TEST 4: Capture Test Image ===
print("\n[TEST 4] Capturing test image...")
ret, frame = camera.read()
if ret:
    print("✓ Image captured successfully!")
else:
    print("✗ ERROR: Cannot capture image!")
    camera.release()
    exit()

print("\n" + "="*50)
print("ALL TESTS PASSED! Starting live test...")
print("="*50)
print("\nControls:")
print("  - Press 'q' to quit")
print("  - Press 's' to save a screenshot")
print("\nShowing live camera with predictions...")
print("="*50 + "\n")

# Get model details once (not in loop - faster!)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === MAIN LOOP: Live Camera + Classification ===
frame_count = 0

try:
    while True:
        # Grab frame from camera
        ret, frame = camera.read()
       
        if not ret:
            print("Camera disconnected!")
            break
       
        frame_count += 1
       
        # Classify every 30 frames (about once per second)
        # This makes it faster - not classifying EVERY frame
        if frame_count % 30 == 0:
           
            # Prepare image for AI
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            small_image = pil_image.resize((224, 224))
           
            # Convert to array
            img_array = np.array(small_image, dtype=np.float32)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
           
            # Run AI prediction
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]
           
            # Get best prediction
            best_index = np.argmax(predictions)
            confidence = predictions[best_index] * 100
            prediction = labels[best_index]
           
            # Print to terminal
            print(f"🎯 {prediction} ({confidence:.1f}%)")
       
        # Draw black box for text
        cv2.rectangle(frame, (5, 5), (635, 90), (0, 0, 0), -1)
       
        # Show prediction on screen
        text = f"{prediction} - {confidence:.0f}%"
        cv2.putText(frame, text, (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
       
        # Show instructions
        cv2.putText(frame, "Press 'q' to quit | 's' to screenshot",
                    (15, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
       
        # Display the window
        cv2.imshow('Camera + AI Test', frame)
       
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
       
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            # Save screenshot
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved: {filename}")

except KeyboardInterrupt:
    print("\n\nStopped by user (Ctrl+C)")

finally:
    # Clean up
    print("\nCleaning up...")
    camera.release()
    cv2.destroyAllWindows()
    print("✓ Done!\n")
