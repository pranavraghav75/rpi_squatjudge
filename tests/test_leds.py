import time
from rpi_ws281x import PixelStrip, Color

LED_COUNT = 16
LED_PIN = 18

strip = PixelStrip(LED_COUNT, LED_PIN)
strip.begin()

colors = [Color(255,0,0), Color(0,255,0), Color(0,0,255)]

while True:
    for c in colors:
        for i in range(LED_COUNT):
            strip.setPixelColor(i, c)
        strip.show()
        time.sleep(0.5)
