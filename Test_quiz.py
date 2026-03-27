# test_every_with_leds.py
# RUN WITH: sudo python3 test_every_with_leds.py

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import time
from picamera2 import Picamera2
from collections import defaultdict
from rpi_ws281x import PixelStrip, Color

print("="*50)
print("SMART BIN — STARTING UP")
print("="*50)

# === LED SETUP ===
LED_COUNT = 180
LED_PIN = 18
LED_BRIGHTNESS = 190

strip = PixelStrip(LED_COUNT, LED_PIN, brightness=LED_BRIGHTNESS)
strip.begin()

LED_GREEN = Color(0, 255, 0)
LED_BLUE  = Color(0, 0, 255)
LED_WHITE = Color(255, 255, 255)
LED_OFF   = Color(0, 0, 0)

COMPOST_ZONE = (0, 60)
RECYCLE_ZONE = (60, 120)
TRASH_ZONE   = (120, 180)
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

def light_up_bin(category):
    if category == "COMPOST":
        light_zone_non_blocking(COMPOST_ZONE, LED_GREEN)
    elif category == "RECYCLE":
        light_zone_non_blocking(RECYCLE_ZONE, LED_BLUE)
    elif category == "TRASH":
        light_zone_non_blocking(TRASH_ZONE, LED_WHITE)

# === WASTE CATEGORY MAPPING ===
def get_bin_category(label):
    label_lower = label.strip().lower()

    COMPOST_EXACT = ['orange peels', 'banana', 'used fiber bowl']

    RECYCLE_EXACT = ['cardboard', 'paper', 'paper bag', 'glass bottle',
                     'glass jar', 'milk carton', 'water bottle', 'soda can']

    if label_lower in COMPOST_EXACT:
        return "COMPOST", (72, 200, 72)

    if label_lower in RECYCLE_EXACT:
        return "RECYCLE", (60, 200, 220)

    # Trash: Used Paper Plate, Used Paper Tray, Used Paper Bowl, Used Tissue,
    # Aluminium Tray, Bubble Wrap, Paper Cup, Plastic Bag, Plastic Container,
    # Plastic Spoon, Plastic Fork, Plastic Wrapper, Ziploc Bag
    return "TRASH", (110, 110, 230)

# === UI HELPERS ===
FONT = cv2.FONT_HERSHEY_SIMPLEX

BG_DARK    = (18, 18, 24)
BG_PANEL   = (30, 32, 42)
ACCENT_GRN = (72, 200, 120)
ACCENT_BLU = (80, 160, 255)
ACCENT_ORG = (60, 165, 255)
WHITE      = (240, 240, 245)
GRAY       = (120, 120, 135)

def filled_bg(frame):
    frame[:] = BG_DARK

def draw_rounded_rect(frame, x, y, w, h, color, radius=16, thickness=-1):
    cv2.rectangle(frame, (x + radius, y), (x + w - radius, y + h), color, thickness)
    cv2.rectangle(frame, (x, y + radius), (x + w, y + h - radius), color, thickness)
    cv2.circle(frame, (x + radius,     y + radius),     radius, color, thickness)
    cv2.circle(frame, (x + w - radius, y + radius),     radius, color, thickness)
    cv2.circle(frame, (x + radius,     y + h - radius), radius, color, thickness)
    cv2.circle(frame, (x + w - radius, y + h - radius), radius, color, thickness)

def centered_text(frame, text, y, scale, color, thickness=2):
    size = cv2.getTextSize(text, FONT, scale, thickness)[0]
    x = (frame.shape[1] - size[0]) // 2
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)

def draw_progress_bar(frame, elapsed, total, y=390):
    W = frame.shape[1]
    bar_w = W - 80
    bx = 40
    bar_h = 8
    cv2.rectangle(frame, (bx, y), (bx + bar_w, y + bar_h), BG_PANEL, -1)
    fill = int(bar_w * min(elapsed / total, 1.0))
    if fill > 4:
        draw_rounded_rect(frame, bx, y, fill, bar_h, ACCENT_BLU, radius=4)

def draw_scan_line(frame, elapsed):
    H, W = frame.shape[:2]
    scan_y = int((elapsed % 2.0) / 2.0 * H)
    overlay = frame.copy()
    cv2.line(overlay, (0, scan_y), (W, scan_y), ACCENT_BLU, 2)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

def draw_corners(frame, color, margin=55, length=28, thickness=3):
    H, W = frame.shape[:2]
    for cx, cy in [(margin, margin), (W-margin, margin),
                   (margin, H-margin), (W-margin, H-margin)]:
        dx = 1 if cx < W // 2 else -1
        dy = 1 if cy < H // 2 else -1
        cv2.line(frame, (cx, cy), (cx + dx*length, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy*length), color, thickness, cv2.LINE_AA)

def draw_bin_card(frame, label, key, x, y, w, h, color):
    bg = tuple(max(0, c // 6) for c in color)
    draw_rounded_rect(frame, x, y, w, h, bg, radius=16)
    draw_rounded_rect(frame, x, y, w, h, color, radius=16, thickness=2)
    kx = x + w // 2
    ky = y + h // 2
    ks = cv2.getTextSize(key, FONT, 1.2, 2)[0]
    cv2.putText(frame, key, (kx - ks[0]//2, ky - 6), FONT, 1.2, color, 2, cv2.LINE_AA)
    ls = cv2.getTextSize(label, FONT, 0.52, 1)[0]
    cv2.putText(frame, label, (kx - ls[0]//2, ky + 28), FONT, 0.52, GRAY, 1, cv2.LINE_AA)

# === LOAD MODEL ===
print("\n[1] Loading AI model...")
interpreter = Interpreter(model_path='model_unquant.tflite')
interpreter.allocate_tensors()
print("✓ Model loaded!")

print("\n[2] Loading labels...")
with open('labels3.txt', 'r') as f:
    labels = []
    for line in f.readlines():
        parts = line.strip().split(' ', 1)   # split on first space only
        label = parts[1] if len(parts) > 1 else parts[0]
        labels.append(label.strip())
print(f"✓ Labels: {labels}")

print("\n[3] Starting camera...")
camera = Picamera2()
camera_config = camera.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
camera.configure(camera_config)
camera.start()
time.sleep(2)
print("✓ Camera ready!\n")
print("SPACE = scan | C/R/T = guess | Q = quit\n")

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IDLE, COLLECTING, QUIZ, SHOWING_RESULT = 0, 1, 2, 3
state = IDLE

collection_start_time = 0
collection_duration   = 5
quiz_start_time  = 0
quiz_duration    = 10
user_guess       = None
quiz_result_msg  = ""
quiz_sub_msg     = ""
quiz_result_color = WHITE

prediction_history = []
confidence_totals  = defaultdict(float)
prediction_counts  = defaultdict(int)

final_decision     = ""
final_confidence   = 0
final_bin_category = ""
final_bin_color    = WHITE

frame_count = 0

try:
    while True:
        raw   = camera.capture_array()
        frame = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        frame_count += 1
        H, W = frame.shape[:2]

        # ── IDLE ──────────────────────────────────────────────────
        if state == IDLE:
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (W,H), (0,0,0), -1)
            cv2.addWeighted(ov, 0.38, frame, 0.62, 0, frame)
            draw_corners(frame, ACCENT_GRN)
            centered_text(frame, "SMART BIN", 155, 1.5, WHITE, 3)
            centered_text(frame, "Hold item up to the camera", 200, 0.62, GRAY, 1)
            bw, bh = 260, 52
            bx = (W - bw) // 2
            draw_rounded_rect(frame, bx, 278, bw, bh, ACCENT_GRN, radius=26)
            centered_text(frame, "SPACE  to scan", 313, 0.72, BG_DARK, 2)

        # ── COLLECTING ────────────────────────────────────────────
        elif state == COLLECTING:
            elapsed   = time.time() - collection_start_time
            remaining = collection_duration - elapsed
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (W,H), (0,0,0), -1)
            cv2.addWeighted(ov, 0.28, frame, 0.72, 0, frame)
            draw_scan_line(frame, elapsed)
            draw_corners(frame, ACCENT_BLU)
            centered_text(frame, "Scanning...", 155, 1.2, WHITE, 2)
            draw_progress_bar(frame, elapsed, collection_duration, y=385)

            if frame_count % 10 == 0:
                img = cv2.resize(raw, (224, 224))
                img = np.array(img, dtype=np.float32) / 255.0
                img = np.expand_dims(img, axis=0)
                interpreter.set_tensor(input_details[0]['index'], img)
                interpreter.invoke()
                preds = interpreter.get_tensor(output_details[0]['index'])[0]
                idx  = np.argmax(preds)
                conf = preds[idx] * 100
                prediction_history.append((idx, conf))
                confidence_totals[idx] += conf
                prediction_counts[idx]  += 1

            if elapsed >= collection_duration:
                best = max(confidence_totals.keys(),
                           key=lambda x: confidence_totals[x] / prediction_counts[x])
                final_decision     = labels[best]
                final_confidence   = confidence_totals[best] / prediction_counts[best]
                final_bin_category, final_bin_color = get_bin_category(final_decision)
                print(f"\nAI decided: {final_decision} → {final_bin_category} (hidden)")
                user_guess = None
                quiz_result_msg = quiz_sub_msg = ""
                quiz_start_time = time.time()
                state = QUIZ

        # ── QUIZ ──────────────────────────────────────────────────
        elif state == QUIZ:
            elapsed_q = time.time() - quiz_start_time
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (W,H), (0,0,0), -1)
            cv2.addWeighted(ov, 0.58, frame, 0.42, 0, frame)

            centered_text(frame, "Where does this go?", 110, 1.0, WHITE, 2)

            btn_w, btn_h = 158, 125
            gap   = 18
            total = btn_w * 3 + gap * 2
            bx0   = (W - total) // 2
            by    = 155

            draw_bin_card(frame, "Compost", "C",
                          bx0,                  by, btn_w, btn_h, (72, 200, 72))
            draw_bin_card(frame, "Recycle", "R",
                          bx0 + btn_w + gap,    by, btn_w, btn_h, (60, 200, 220))
            draw_bin_card(frame, "Trash",   "T",
                          bx0 + (btn_w+gap)*2,  by, btn_w, btn_h, (120, 110, 240))

            centered_text(frame, "Press  C  R  or  T", 355, 0.62, GRAY, 1)

            if elapsed_q >= quiz_duration and user_guess is None:
                user_guess        = "NONE"
                quiz_result_msg   = "Time's up!"
                quiz_sub_msg      = f"It goes in  {final_bin_category.capitalize()}"
                quiz_result_color = ACCENT_ORG
                print(f"No answer. Answer: {final_bin_category}")
                light_up_bin(final_bin_category)
                state = SHOWING_RESULT

        # ── RESULT ────────────────────────────────────────────────
        elif state == SHOWING_RESULT:
            filled_bg(frame)
            panel_bg = tuple(max(0, c // 5) for c in final_bin_color)
            draw_rounded_rect(frame, 36, 88, W-72, 210, panel_bg, radius=22)
            draw_rounded_rect(frame, 36, 88, W-72, 210, final_bin_color, radius=22, thickness=2)
            centered_text(frame, quiz_result_msg, 178, 1.15, quiz_result_color, 2)
            centered_text(frame, quiz_sub_msg,    228, 0.72, WHITE, 1)

            chip = f"{final_decision}   {final_confidence:.0f}% confident"
            cs   = cv2.getTextSize(chip, FONT, 0.52, 1)[0]
            cx   = (W - cs[0] - 28) // 2
            draw_rounded_rect(frame, cx, 335, cs[0]+28, 34, BG_PANEL, radius=12)
            cv2.putText(frame, chip, (cx+14, 358), FONT, 0.52, GRAY, 1, cv2.LINE_AA)

            centered_text(frame, "SPACE  for next item", 430, 0.62, GRAY, 1)

        # ── LED AUTO-OFF ──────────────────────────────────────────
        if led_active_until != 0 and time.time() > led_active_until:
            clear_strip()
            led_active_until = 0

        cv2.imshow("Smart Bin", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # Escape
            break
        elif key == ord(' '):
            if state == IDLE:
                state = COLLECTING
                collection_start_time = time.time()
                prediction_history = []
                confidence_totals  = defaultdict(float)
                prediction_counts  = defaultdict(int)
                frame_count = 0
                print("Scanning...")
            elif state == SHOWING_RESULT:
                state = IDLE
        elif state == QUIZ and user_guess is None:
            guess_map = {ord('c'): "COMPOST", ord('r'): "RECYCLE", ord('t'): "TRASH"}
            if key in guess_map:
                user_guess = guess_map[key]
                print(f"Guess: {user_guess}  |  AI: {final_bin_category}")
                if user_guess == final_bin_category:
                    quiz_result_msg   = "You got it!"
                    quiz_sub_msg      = f"It is {final_bin_category.capitalize()}  —  nice one!"
                    quiz_result_color = ACCENT_GRN
                else:
                    quiz_result_msg   = "Not quite!"
                    quiz_sub_msg      = f"It's {final_bin_category.capitalize()}, not {user_guess.capitalize()}"
                    quiz_result_color = ACCENT_ORG
                light_up_bin(final_bin_category)
                state = SHOWING_RESULT

except KeyboardInterrupt:
    print("Stopped")

finally:
    clear_strip()
    camera.stop()
    cv2.destroyAllWindows()
    print("Done!")
