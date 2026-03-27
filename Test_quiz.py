# test_every_with_leds.py
# RUN WITH: sudo python3 test_every_with_leds.py

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
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

def light_up_bin(category):
    if category == "COMPOST":
        light_zone_non_blocking(COMPOST_ZONE, LED_GREEN)
    elif category == "RECYCLE":
        light_zone_non_blocking(RECYCLE_ZONE, LED_BLUE)
    elif category == "TRASH":
        light_zone_non_blocking(TRASH_ZONE, LED_WHITE)

# === LOAD MODEL ===
print("\n[TEST 1] Loading AI model...")
interpreter = Interpreter(model_path='model_unquant.tflite')
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
IDLE, COLLECTING, QUIZ, SHOWING_RESULT = 0, 1, 2, 3
state = IDLE

collection_start_time = 0
collection_duration = 5

quiz_start_time = 0
quiz_duration = 10
user_guess = None
quiz_result_msg = ""
quiz_result_color = (255, 255, 255)

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

                print(f"\nAI decided: {final_decision} → {final_bin_category} (hidden until you guess)")

                # Move to quiz state — reset guess tracking
                user_guess = None
                quiz_result_msg = ""
                quiz_start_time = time.time()
                state = QUIZ

        # === QUIZ ===
        elif state == QUIZ:
            elapsed = time.time() - quiz_start_time
            remaining = quiz_duration - elapsed

            cv2.putText(frame, "Where does this go?", (100, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, "C = Compost  R = Recycle  T = Trash", (60, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, f"Time left: {remaining:.1f}s", (200, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

            # Time ran out with no answer
            if remaining <= 0 and user_guess is None:
                user_guess = "NONE"
                quiz_result_msg = f"Time's up! It is {final_bin_category.capitalize()}"
                quiz_result_color = (0, 165, 255)
                print(f"No answer given. Answer was: {final_bin_category}")
                light_up_bin(final_bin_category)
                state = SHOWING_RESULT

        # === SHOW RESULT ===
        elif state == SHOWING_RESULT:
            # Main result line
            cv2.putText(frame, quiz_result_msg, (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, quiz_result_color, 2)
            # Show item name below
            cv2.putText(frame, f"({final_decision}, {final_confidence:.0f}% confident)",
                        (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            cv2.putText(frame, "Press SPACE for next", (140, 310),
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

        # === QUIZ INPUT (C / R / T) ===
        elif state == QUIZ and user_guess is None:
            guess_map = {
                ord('c'): "COMPOST",
                ord('r'): "RECYCLE",
                ord('t'): "TRASH",
            }
            if key in guess_map:
                user_guess = guess_map[key]
                print(f"You guessed: {user_guess} | answer: {final_bin_category}")

                if user_guess == final_bin_category:
                    quiz_result_msg = f"Correct! It is {final_bin_category.capitalize()} :)"
                    quiz_result_color = (0, 220, 0)
                    print("Correct!")
                else:
                    quiz_result_msg = f"Almost! It's {final_bin_category.capitalize()}, not {user_guess.capitalize()}"
                    quiz_result_color = (0, 165, 255)
                    print(f"It was {final_bin_category}.")

                light_up_bin(final_bin_category)
                state = SHOWING_RESULT

except KeyboardInterrupt:
    print("Stopped")

finally:
    clear_strip()
    camera.stop()
    cv2.destroyAllWindows()
    print("Done!")
