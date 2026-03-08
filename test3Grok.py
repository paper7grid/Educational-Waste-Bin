# test_everything.py
# 2026 - Natural color fix for Pi Camera + TFLite classification
# Goal: neutral / normal looking video (no blue, no orange cast)

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time
from picamera2 import Picamera2

print("=" * 50)
print("Pi Camera + AI Classification - NATURAL COLOR MODE")
print("Use + / - to tune red gain, [ / ] for blue gain")
print("=" * 50)

# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────

MODEL_FILE      = "model_unquant.tflite"          # ← Rename your file to this OR change the string
LABELS_FILE     = "labels.txt"
INPUT_SIZE      = (224, 224)
DISPLAY_SIZE    = (800, 600)



CLASSIFY_EVERY  = 30

# Starting gains — tune these live!
# (red_gain, blue_gain) — higher red = warmer, higher blue = cooler
START_RED_GAIN  = 1.80
START_BLUE_GAIN = 1.40

# ────────────────────────────────────────────────
#  Load model
# ────────────────────────────────────────────────

print("\n[1] Loading TFLite model...")
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()
    print("✓ Model loaded")
except Exception as e:
    print(f"✗ Failed to load '{MODEL_FILE}': {e}")
    print("   → Check file name, path, or try: ls -l *.tflite")
    exit(1)

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ────────────────────────────────────────────────
#  Load labels
# ────────────────────────────────────────────────

print("\n[2] Loading labels...")
try:
    with open(LABELS_FILE, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"✓ {len(labels)} labels")
except Exception as e:
    print(f"✗ Labels failed: {e}")
    exit(1)

# ────────────────────────────────────────────────
#  Camera setup
# ────────────────────────────────────────────────

print("\n[3] Starting Pi Camera...")
camera = Picamera2()

config = camera.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}   # ← RGB888 avoids swap issues
)
camera.configure(config)
camera.start()

# Manual white balance - start neutral-ish for daylight / indoor mixed
camera.set_controls({
    "AwbEnable": False,
    "ColourGains": (START_RED_GAIN, START_BLUE_GAIN),
    "AeEnable": True,
    "Brightness": 0.0,
    "Contrast": 1.15,
    "Saturation": 1.10               # slight boost — looks more natural
})

time.sleep(1.5)
print("✓ Camera running — manual WB")

print("\n" + "="*60)
print("  q       → quit")
print("  s       → save screenshot")
print("  + / =   → more red / warmer")
print("  - / _   → less red / cooler")
print("  [       → more blue")
print("  ]       → less blue")
print("="*60 + "\n")

# ────────────────────────────────────────────────
#  Main loop
# ────────────────────────────────────────────────

frame_count = 0
red_gain   = START_RED_GAIN
blue_gain  = START_BLUE_GAIN
prediction = "Starting..."
confidence = 0.0

try:
    while True:
        frame_rgb = camera.capture_array()           # already RGB !

        # Very light post-process — optional / mild
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # frame_bgr = cv2.convertScaleAbs(frame_bgr, alpha=1.05, beta=5)  # optional tiny brightness/contrast

        # Display resize
        display = cv2.resize(frame_bgr, DISPLAY_SIZE)

        frame_count += 1

        # ── Inference ─────────────────────────────────
        if frame_count % CLASSIFY_EVERY == 0:
            pil_img = Image.fromarray(frame_rgb)
            resized = pil_img.resize(INPUT_SIZE)
            
            arr = np.array(resized, dtype=np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)

            interpreter.set_tensor(input_details[0]['index'], arr)
            interpreter.invoke()
            
            probs = interpreter.get_tensor(output_details[0]['index'])[0]
            idx = np.argmax(probs)
            confidence = probs[idx] * 100
            prediction = labels[idx]

            print(f"  → {prediction:20} {confidence:5.1f}%   "
                  f"(R:{red_gain:.2f}  B:{blue_gain:.2f})")

        # ── Overlay ───────────────────────────────────
        cv2.rectangle(display, (10, 10), (DISPLAY_SIZE[0]-10, 120), (0,0,0), -1)
        
        cv2.putText(display, prediction,
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 2)
        cv2.putText(display, f"Conf: {confidence:.0f}%",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 255), 2)
        cv2.putText(display, f"R gain: {red_gain:.2f}   B gain: {blue_gain:.2f}",
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 180), 1)

        cv2.imshow("Camera + Classification", display)

        # ── Keys ──────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = int(time.time())
            fn = f"capture_{ts}.jpg"
            cv2.imwrite(fn, frame_bgr)
            print(f"  Saved {fn}")
        elif key in (ord('+'), ord('=')):
            red_gain = min(4.0, red_gain + 0.08)
            camera.set_controls({"ColourGains": (red_gain, blue_gain)})
            print(f"  Red gain → {red_gain:.2f}")
        elif key in (ord('-'), ord('_')):
            red_gain = max(0.5, red_gain - 0.08)
            camera.set_controls({"ColourGains": (red_gain, blue_gain)})
            print(f"  Red gain → {red_gain:.2f}")
        elif key == ord('['):
            blue_gain = min(4.0, blue_gain + 0.06)
            camera.set_controls({"ColourGains": (red_gain, blue_gain)})
            print(f"  Blue gain → {blue_gain:.2f}")
        elif key == ord(']'):
            blue_gain = max(0.5, blue_gain - 0.06)
            camera.set_controls({"ColourGains": (red_gain, blue_gain)})
            print(f"  Blue gain → {blue_gain:.2f}")

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    camera.stop()
    cv2.destroyAllWindows()
    print("Done.")
