"""
SIMPLE SSH CONTROL SCRIPT
NOTE: BEFORE RUNNING INITIALIZE VIRTUAL ENVIRONMENT (myenv) with 'source myenv/bin/activate'
"""

import numpy as np
import time
import board
import adafruit_adxl34x
from adafruit_motorkit import MotorKit
import sys
import termios
import tty
import select

motors = MotorKit(i2c=board.I2C())
    
# Set throttle for the selected motor
motor_mapping = {
    1: motors.motor1,
    2: motors.motor2,
    3: motors.motor3,
    4: motors.motor4
}

# Pick a motor and set the throttle
def _set_throttle(motor, throttle):
    # Check if throttle and motor are in range
    if throttle < -1:
        throttle = -1
    elif throttle > 1:
        throttle = 1
    
    if motor not in motor_mapping:
        print("Motor is not in range. It must be between 1 and 4.")

    motor_mapping[motor].throttle = throttle

# Stop all motors
def _stop_all_motors():
    for i in range(1, 5):
        motor_mapping[i].throttle = 0


def mix(steer: float, throttle: float) -> None:

    """
    creates a cosine mix based on throttle and steer
    """
    throttle = np.clip(throttle, a_min=-1, a_max=1)
    steer = np.clip(steer, a_min=-1, a_max=1)

    a_l = 1/2 * (np.cos( (2*np.pi*(steer+1)) / 4 ) + 1) * throttle 
    a_r = 1/2 * (np.cos( (2*np.pi*(steer-1)) / 4 ) + 1) * throttle

    _set_throttle(2, a_r)
    _set_throttle(4, a_r)
    _set_throttle(1, a_l)
    _set_throttle(3, a_l)



def drive():
    fd = sys.stdin.fileno()
    original_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)

        steer = 0
        throttle = 0


        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:  # Check if input is available
                char = sys.stdin.read(1)

                if char == '\x1b':  # Escape sequence detected
                    seq = sys.stdin.read(2)  # Read the next two characters

                    if seq == '[A':
                        throttle = 1

                    elif seq == '[B':
                        throttle = -1

                    elif seq == '[C':
                        steer = -1

                    elif seq == '[D':
                        steer = 1

                elif char == 'q':  # Exit on 'q'
                    break

                elif char == ' ':
                    throttle = 0

                else:
                    pass

            else:
                steer = 0

            mix(steer, throttle)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)


# Run the function
if __name__ == "__main__":

    print("select throttle with up and down arrows (don't hold these down)")
    print("stop motors with space bar")
    print("steer with arrow keys (you can hold these down)")
    print("press q to quit")

    drive()
    _stop_all_motors()

