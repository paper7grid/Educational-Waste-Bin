# sequential_zones.py
# RUN WITH: sudo python3 sequential_zones.py

from rpi_ws281x import PixelStrip, Color
import time
import sys
import select

# Config
LED_COUNT = 180
LED_PIN = 18
LED_BRIGHTNESS = 190

strip = PixelStrip(LED_COUNT, LED_PIN, brightness=LED_BRIGHTNESS)
strip.begin()

# Colors (GRB format)
GREEN = Color(0, 255, 0)
BLUE = Color(0, 0, 255)
WHITE = Color(255, 255, 255)
OFF = Color(0, 0, 0)

def clear_strip():
    for i in range(LED_COUNT):
        strip.setPixelColor(i, OFF)
    strip.show()

def light_range(start, end, color):
    clear_strip()
    for i in range(start, end):
        strip.setPixelColor(i, color)
    strip.show()

def wait_or_enter(seconds):
    print(f"Waiting {seconds}s or press Enter...")
    i, o, e = select.select([sys.stdin], [], [], seconds)
    if i:
        sys.stdin.readline()

# Run sequence
print("Starting LED sequence...")

# Zone 1: 0–19 GREEN
light_range(0, 60, GREEN)
wait_or_enter(3)

# Zone 2: 20–39 BLUE
light_range(61, 120, BLUE)
wait_or_enter(3)

# Zone 3: 40–59 WHITE
light_range(121, 180, WHITE)
wait_or_enter(1)

# Turn everything off at end
clear_strip()

print("Done!")
