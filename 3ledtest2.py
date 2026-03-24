# test_leds_only.py
# Test LED strips only - RUN WITH SUDO!

import time
from rpi_ws281x import PixelStrip, Color

print("="*50)
print("LED STRIP TEST")
print("="*50)

# LED Configuration
LED_COUNT = 60              # LEDs per strip
LED_BRIGHTNESS = 190        # 75% brightness
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_INVERT = False

# GPIO pins
LED_PINS = [24] #13]     # Compost, Recycling, Trash

# Colors (GRB format!)
GREEN = Color(255, 0, 0)    # Compost
BLUE = Color(0, 0, 255)     # Recycling
GRAY = Color(50, 50, 50)    # Trash
OFF = Color(0, 0, 0)

# Create LED strip objects
print("\nInitializing LED strips...")
led_strips = []
for i, pin in enumerate(LED_PINS):
    print(f"  Strip {i+1} on GPIO {pin}...")
    strip = PixelStrip(LED_COUNT, pin, LED_FREQ_HZ, LED_DMA,
                       LED_INVERT, LED_BRIGHTNESS)
    strip.begin()
    led_strips.append(strip)

print("✓ All strips initialized!")

# Test function
def light_strip(strip_index, color, duration=2):
    """Light up one strip"""
    strip = led_strips[strip_index]
   
    # Turn on
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, color)
    strip.show()
   
    time.sleep(duration)
   
    # Turn off
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, OFF)
    strip.show()

# Run tests
print("\n" + "="*50)
print("TESTING EACH STRIP")
print("="*50)

print("\nTest 1: Compost strip (GREEN)")
light_strip(0, GREEN, 3)

print("Test 2: Recycling strip (BLUE)")
light_strip(1, BLUE, 3)

print("Test 3: Trash strip (GRAY)")
light_strip(2, GRAY, 3)

print("\nTest 4: All strips together (WHITE)")
for strip in led_strips:
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, Color(255, 255, 255))
    strip.show()

time.sleep(3)

# Turn all off
print("\nTurning off all LEDs...")
for strip in led_strips:
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, OFF)
    strip.show()

print("✓ Test complete!")
