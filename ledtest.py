
# test_one_led.py
# Super simple test for ONE LED strip

import time
from rpi_ws281x import PixelStrip, Color

print("="*50)
print("LED STRIP TEST - One Strip")
print("="*50)

# Configuration
LED_COUNT = 60              # Your strips have 60 LEDs each
LED_PIN = 18                # GPIO 18 (Pin 12)
LED_BRIGHTNESS = 190        # 75% brightness (safe)
LED_FREQ_HZ = 800000
LED_DMA = 10

# Create strip object
print("\nInitializing LED strip...")
strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA,
                   False, LED_BRIGHTNESS)
strip.begin()
print("✓ Strip initialized!")

# Helper function
def set_all_color(color, strip_obj):
    """Set all LEDs to one color"""
    for i in range(strip_obj.numPixels()):
        strip_obj.setPixelColor(i, color)
    strip_obj.show()

def turn_off(strip_obj):
    """Turn off all LEDs"""
    set_all_color(Color(0, 0, 0), strip_obj)

# === COLOR DEFINITIONS ===
# NOTE: WS2812B uses GRB order (Green, Red, Blue), NOT RGB!
GREEN = Color(255, 0, 0)      # For Compost
BLUE = Color(0, 0, 255)       # For Recycling  
WHITE = Color(100, 100, 100)  # For Trash (gray/white)
OFF = Color(0, 0, 0)

print("\n" + "="*50)
print("TESTING COLORS")
print("="*50)

try:
    # Test 1: GREEN
    print("\nTest 1: GREEN (for Compost)")
    print("  All LEDs should turn GREEN for 3 seconds...")
    input("  Press ENTER to start test 1...")
    set_all_color(GREEN, strip)
    time.sleep(3)
    turn_off(strip)
    print("  ✓ Test 1 complete")
    time.sleep(1)
   
    # Test 2: BLUE
    print("\nTest 2: BLUE (for Recycling)")
    print("  All LEDs should turn BLUE for 3 seconds...")
    input("  Press ENTER to start test 2...")
    set_all_color(BLUE, strip)
    time.sleep(3)
    turn_off(strip)
    print("  ✓ Test 2 complete")
    time.sleep(1)
   
    # Test 3: WHITE/GRAY
    print("\nTest 3: WHITE/GRAY (for Trash)")
    print("  All LEDs should turn WHITE for 3 seconds...")
    input("  Press ENTER to start test 3...")
    set_all_color(WHITE, strip)
    time.sleep(3)
    turn_off(strip)
    print("  ✓ Test 3 complete")
   
    print("\n" + "="*50)
    print("ALL TESTS COMPLETE!")
    print("="*50)
   
    # Ask about results
    print("\nDid the LEDs light up?")
    print("  1. YES - All colors looked good!")
    print("  2. NO - Nothing happened")
    print("  3. WRONG COLORS - Colors were mixed up")
   
    choice = input("\nEnter 1, 2, or 3: ")
   
    if choice == "1":
        print("\n🎉 SUCCESS! Your LED strip is working perfectly!")
        print("Ready to test the other 2 strips!")
    elif choice == "2":
        print("\n⚠️ TROUBLESHOOTING:")
        print("  - Check power supply is plugged in and turned on")
        print("  - Check data wire is in Pin 12 (GPIO 18)")
        print("  - Check ground wire is in Pin 6")
        print("  - Make sure you ran with 'sudo'")
    elif choice == "3":
        print("\n⚠️ Colors are swapped - that's OK!")
        print("  WS2812B LEDs use GRB order, not RGB")
        print("  We can fix this in the code")

except KeyboardInterrupt:
    print("\n\nStopped by user")

finally:
    # Always turn off when done
    turn_off(strip)
    print("\n✓ LEDs turned off")
    print("Done!")
