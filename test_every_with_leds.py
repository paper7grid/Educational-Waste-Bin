# test_every_with_leds.py
# RUN WITH: sudo python3 test_every_with_leds.py

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
from picamera2 import Picamera2
from collections import defaultdict
from rpi_ws281x import PixelStrip, Color

print("="*50)
print("TESTING CAMERA + AI MODEL + LED SYSTEM")
print("="*50)

# === LED SETUP ===
LED_COUNT = 180
LED_PIN = 18
LED_BRIGHTNESS = 190

strip = PixelStrip(LED_COUNT, LED_PIN, brightness=LED_BRIGHTNESS)
strip.begin()

LED_GREEN = Color(0, 255, 0)
LED_BLUE = Color(0, 0, 255)
LED_WHITE = Color(255, 255, 255)
LED_OFF = Color(0, 0, 0)

COMPOST_ZONE = (0, 60)
RECYCLE_ZONE = (60, 120)
TRASH_ZONE = (120, 180)

led_active_until = 0

def clear_strip():
    for i in range(LED_COUNT):
        strip.setPixelColor(i, LED_OFF)
    strip.show()

def light_zone_non_blocking(zone, color, duration=5):
    global led_active_until
    start, end = zone

    clear_strip()

    for i in range(start, end):
        strip.setPixelColor(i, color)
    strip.show()

    led_active_until = time.time() + duration

# === WASTE CATEGORY MAPPING ===
def get_bin_category(label):
    label_lower = label.lower()
    if 'banana' in label_lower or 'orange peel' in label_lower:
        return "COMPOST", (34, 139, 34)
    elif 'paperplate' in label_lower or 'paper plate' in label_lower:
        return "TRASH", (0, 0, 200)
    elif 'cardboard' in label_lower:
        return "RECYCLE", (200, 130, 0)
    else:
        return "UNKNOWN", (128, 128, 128)

# === LOAD MODEL ===
print("\n[TEST 1] Loading AI model...")
interpreter = tf.lite.Interpreter(model_path='model_unquant.tflite')
interpreter.allocate_tensors()
print("✓ Model loaded!")

# === LOAD LABELS ===
print("\n[TEST 2] Loading labels...")
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]
print(f"✓ Labels: {labels}")

# === CAMERA SETUP ===
print("\n[TEST 3] Starting camera...")
camera = Picamera2()
camera_config = camera.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
camera.configure(camera_config)
camera.start()
time.sleep(2)
print("✓ Camera started!")

print("\nPress SPACE to analyze | Q to quit\n")

# Model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# States
IDLE, COLLECTING, SHOWING_RESULT = 0, 1, 2
state = IDLE

collection_start_time = 0
collection_duration = 5

prediction_history = []
confidence_totals = defaultdict(float)
prediction_counts = defaultdict(int)

final_decision = ""
final_confidence = 0
final_bin_category = ""
final_bin_color = (255, 255, 255)

frame_count = 0

try:
    while True:
        frame = camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_count += 1

        # === IDLE ===
        if state == IDLE:
            cv2.putText(frame, "READY - Press SPACE", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # === COLLECTING ===
        elif state == COLLECTING:
            elapsed = time.time() - collection_start_time
            remaining = collection_duration - elapsed

            if frame_count % 10 == 0:
                img = cv2.resize(frame, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.array(img, dtype=np.float32) / 255.0
                img = np.expand_dims(img, axis=0)

                interpreter.set_tensor(input_details[0]['index'], img)
                interpreter.invoke()
                preds = interpreter.get_tensor(output_details[0]['index'])[0]

                idx = np.argmax(preds)
                conf = preds[idx] * 100

                prediction_history.append((idx, conf))
                confidence_totals[idx] += conf
                prediction_counts[idx] += 1

            cv2.putText(frame, f"Analyzing: {remaining:.1f}s", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            if elapsed >= collection_duration:
                best = max(confidence_totals.keys(),
                           key=lambda x: confidence_totals[x] / prediction_counts[x])

                final_decision = labels[best]
                final_confidence = confidence_totals[best] / prediction_counts[best]

                final_bin_category, final_bin_color = get_bin_category(final_decision)

                print(f"\nFINAL: {final_decision} → {final_bin_category}")

                # 🔥 LED TRIGGER
                if final_bin_category == "COMPOST":
                    light_zone_non_blocking(COMPOST_ZONE, LED_GREEN)
                elif final_bin_category == "RECYCLE":
                    light_zone_non_blocking(RECYCLE_ZONE, LED_BLUE)
                elif final_bin_category == "TRASH":
                    light_zone_non_blocking(TRASH_ZONE, LED_WHITE)

                state = SHOWING_RESULT

        # === SHOW RESULT ===
        elif state == SHOWING_RESULT:
            cv2.putText(frame, f"{final_bin_category}", (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, final_bin_color, 3)
            cv2.putText(frame, "Press SPACE for next", (140, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # === AUTO TURN OFF LED ===
        if led_active_until != 0 and time.time() > led_active_until:
            clear_strip()
            led_active_until = 0

        cv2.imshow("Smart Bin", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if state == IDLE:
                state = COLLECTING
                collection_start_time = time.time()
                prediction_history = []
                confidence_totals = defaultdict(float)
                prediction_counts = defaultdict(int)
                print("Analyzing...")
            elif state == SHOWING_RESULT:
                state = IDLE

except KeyboardInterrupt:
    print("Stopped")

finally:
    clear_strip()
    camera.stop()
    cv2.destroyAllWindows()
    print("Done!")
