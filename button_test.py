from gpiozero import Button

button = Button(2, bounce_time=0.1)

while True:
    button.wait_for_press()
    print("Button Pressed!")
    button.wait_for_release()
