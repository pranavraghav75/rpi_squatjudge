from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16)
pan = 0
tilt = 1

for i in range(0, 180, 10):
    kit.servo[pan].angle = i
    kit.servo[tilt].angle = 180 - i
    time.sleep(0.1)
