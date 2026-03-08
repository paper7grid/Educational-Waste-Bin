# test_everything.py
# AGGRESSIVE blue tint fix

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
from picamera2 import Picamera2

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

# Function to fix blue tint
def fix_blue_tint(frame):
    """Remove blue tint by adjusting BGR channels"""
    frame_float = frame.astype(np.float32)
    
    # Adjust color channels (BGR format)
    frame_float[:, :, 0] = frame_float[:, :, 0] * 1.4  # Boost Blue (less)
    frame_float[:, :, 1] = frame_float[:, :, 1] * 1.2  # Boost Green
    frame_float[:, :, 2] = frame_float[:, :, 2] * 0.6  # Reduce Red (actually blue in BGR)
    
    # Wait, OpenCV uses BGR, so let me fix this:
    # B, G, R = 0, 1, 2
    frame_float[:, :, 2] = frame_float[:, :, 2] * 1.5  # Boost Red
    frame_float[:, :, 1] = frame_float[:, :, 1] * 1.1  # Slight Green
    frame_float[:, :, 0] = frame_float[:, :, 0] * 0.6  # Reduce Blue
    
    frame_float = np.clip(frame_float, 0, 255)
    return frame_float.astype(np.uint8)

# === TEST 3: Start Camera - MANUAL COLOR ===
print("\n[TEST 3] Starting camera...")
camera = Picamera2()

camera_config = camera.create_preview_configuration(
    main={"size": (1280, 720), "format": "XBGR8888"}
)

camera.configure(camera_config)
camera.start()

# Turn OFF auto white balance and set manual gains
camera.set_controls({
    "AwbEnable": False,
    "ColourGains": (2.8, 1.0),  # High red, low blue
    "AeEnable": True,
    "Brightness": 0.0,
    "Contrast": 1.3
})

time.sleep(2)
print("✓ Camera started with manual color correction!")

print("\n" + "="*50)
print("Controls:")
print("  - Press 'q' to quit")
print("  - Press 's' to save screenshot")
print("  - Press '+' to reduce blue more")
print("  - Press '-' to reduce blue less")
print("="*50 + "\n")

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Main loop
frame_count = 0
prediction = "Loading..."
confidence = 0
blue_reduction = 0.6  # Adjustable blue reduction

try:
    while True:
        # Capture frame
        frame = camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # APPLY SOFTWARE COLOR CORRECTION
        frame_float = frame.astype(np.float32)
        frame_float[:, :, 2] = frame_float[:, :, 2] * 1.5   # Red
        frame_float[:, :, 1] = frame_float[:, :, 1] * 1.1   # Green  
        frame_float[:, :, 0] = frame_float[:, :, 0] * blue_reduction  # Blue
        frame = np.clip(frame_float, 0, 255).astype(np.uint8)
        
        # Resize for display
        display_frame = cv2.resize(frame, (800, 600))
        
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
            
            print(f"\n🎯 {prediction} ({confidence:.1f}%)")
        
        # Draw on screen
        cv2.rectangle(display_frame, (5, 5), (795, 100), (0, 0, 0), -1)
        cv2.putText(display_frame, f"{prediction}", (15, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Confidence: {confidence:.0f}%", (15, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Camera Test', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'screenshot_{int(time.time())}.jpg'
            cv2.imwrite(filename, frame)
            print(f"📸 Saved: {filename}")
        elif key == ord('+') or key == ord('='):
            blue_reduction = max(0.3, blue_reduction - 0.05)
            print(f"🎨 Blue reduction: {blue_reduction:.2f}")
        elif key == ord('-') or key == ord('_'):
            blue_reduction = min(1.0, blue_reduction + 0.05)
            print(f"🎨 Blue reduction: {blue_reduction:.2f}")

except KeyboardInterrupt:
    print("\nStopped")

finally:
    camera.stop()
    cv2.destroyAllWindows()
    print("Done!")
