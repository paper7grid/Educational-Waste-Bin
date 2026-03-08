# test_every.py
# Test camera + AI model with 5-second smart decision making

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
from picamera2 import Picamera2
from collections import defaultdict

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
camera = Picamera2()
camera_config = camera.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
camera.configure(camera_config)
camera.set_controls({
    "AwbEnable": True,
    "AwbMode": 0,
    "AeEnable": True,
    "Brightness": 0.0,
    "Contrast": 1.0
})
camera.start()
time.sleep(2)
print("✓ Camera started!")

print("\n" + "="*50)
print("HOW IT WORKS:")
print("  1. Press SPACEBAR to start analyzing")
print("  2. Hold item steady for 5 seconds")
print("  3. See final decision")
print("  4. Press SPACEBAR for next item")
print("  5. Press 'q' to quit")
print("="*50 + "\n")

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# State management
IDLE = 0
COLLECTING = 1
SHOWING_RESULT = 2

state = IDLE
collection_start_time = 0
collection_duration = 5  # 5 seconds

# Data collection
prediction_history = []
confidence_totals = defaultdict(float)
prediction_counts = defaultdict(int)

final_decision = ""
final_confidence = 0

frame_count = 0

try:
    while True:
        # Capture frame
        frame = camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frame_count += 1
        
        # === STATE: IDLE - Waiting to start ===
        if state == IDLE:
            # Draw instructions
            cv2.rectangle(frame, (50, 150), (590, 330), (0, 0, 0), -1)
            cv2.rectangle(frame, (50, 150), (590, 330), (0, 255, 0), 3)
            
            cv2.putText(frame, "READY", (220, 210), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, "Show item to camera", (100, 260), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Press SPACEBAR to start", (90, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # === STATE: COLLECTING - Analyzing for 5 seconds ===
        elif state == COLLECTING:
            elapsed = time.time() - collection_start_time
            time_remaining = collection_duration - elapsed
            
            # Classify every 10 frames
            if frame_count % 10 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                small_image = pil_image.resize((224, 224))
                
                img_array = np.array(small_image, dtype=np.float32)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])[0]
                
                # Get top prediction
                best_index = np.argmax(predictions)
                best_confidence = predictions[best_index] * 100
                
                # Store prediction
                prediction_history.append((best_index, best_confidence))
                confidence_totals[best_index] += best_confidence
                prediction_counts[best_index] += 1
                
                # Show current prediction (small text)
                current_label = labels[best_index]
                cv2.putText(frame, f"Seeing: {current_label}", (10, 450), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Draw countdown box
            cv2.rectangle(frame, (150, 120), (490, 240), (0, 0, 0), -1)
            cv2.rectangle(frame, (150, 120), (490, 240), (255, 165, 0), 3)
            
            cv2.putText(frame, "ANALYZING...", (170, 160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 2)
            cv2.putText(frame, f"{time_remaining:.1f}s", (250, 210), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Progress bar
            progress = elapsed / collection_duration
            bar_width = int(290 * progress)
            cv2.rectangle(frame, (175, 220), (465, 230), (100, 100, 100), -1)
            cv2.rectangle(frame, (175, 220), (175 + bar_width, 230), (0, 255, 0), -1)
            
            # Check if time is up
            if elapsed >= collection_duration:
                # Make final decision
                if prediction_counts:
                    # Find label with highest average confidence
                    best_label = max(confidence_totals.keys(), 
                                   key=lambda x: confidence_totals[x] / prediction_counts[x])
                    
                    final_decision = labels[best_label]
                    final_confidence = confidence_totals[best_label] / prediction_counts[best_label]
                    
                    # Print detailed analysis
                    print("\n" + "="*50)
                    print("ANALYSIS COMPLETE!")
                    print("="*50)
                    print(f"\nCollected {len(prediction_history)} predictions over {collection_duration} seconds\n")
                    print("Results by frequency:")
                    for label_idx in sorted(prediction_counts.keys(), 
                                          key=lambda x: prediction_counts[x], reverse=True):
                        count = prediction_counts[label_idx]
                        avg_conf = confidence_totals[label_idx] / count
                        percentage = (count / len(prediction_history)) * 100
                        print(f"  {labels[label_idx]}: {count} times ({percentage:.1f}%) - Avg: {avg_conf:.1f}%")
                    
                    print(f"\n🎯 FINAL DECISION: {final_decision}")
                    print(f"📊 Confidence: {final_confidence:.1f}%")
                    print("="*50 + "\n")
                
                state = SHOWING_RESULT
        
        # === STATE: SHOWING_RESULT - Display final decision ===
        elif state == SHOWING_RESULT:
            # Big result box
            cv2.rectangle(frame, (30, 80), (610, 400), (0, 0, 0), -1)
            cv2.rectangle(frame, (30, 80), (610, 400), (0, 255, 0), 5)
            
            cv2.putText(frame, "RESULT", (230, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Item name (handle long text)
            font_scale = 1.2 if len(final_decision) < 20 else 0.9
            cv2.putText(frame, final_decision, (60, 230), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)
            
            cv2.putText(frame, f"Confidence: {final_confidence:.1f}%", (150, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            cv2.putText(frame, "Press SPACEBAR for next", (100, 360), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Show frame
        cv2.imshow('Smart Waste Classifier', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar
            if state == IDLE:
                # Start collection
                state = COLLECTING
                collection_start_time = time.time()
                prediction_history = []
                confidence_totals = defaultdict(float)
                prediction_counts = defaultdict(int)
                print("\n📸 Started analyzing... hold item steady")
            elif state == SHOWING_RESULT:
                # Reset to idle
                state = IDLE
                print("\n✅ Ready for next item")

except KeyboardInterrupt:
    print("\nStopped")

finally:
    camera.stop()
    cv2.destroyAllWindows()
    print("Done!")




