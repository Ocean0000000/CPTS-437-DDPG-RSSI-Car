"""
This script tracks the Received Signal Strength Indicator (RSSI) of a Bluetooth device and estimates the distance to the device based on the RSSI values. 
It uses a moving average window to smooth the RSSI values and provides a more stable distance estimation.
Functions:
    get_rssi_bluetooth(mac_address):
        Retrieves the RSSI value for a given Bluetooth device MAC address using the `hcitool` command.
    rssi_to_distance(rssi):
        Converts an RSSI value to an estimated distance using the log-distance path loss model.
Constants:
    RSSI_0 (int): The RSSI value at 1 meter distance (to be calibrated).
    N (float): The path loss exponent (to be calibrated).
    MOVING_AVERAGE_WINDOW (int): The window size for the moving average of RSSI values.
Usage:
    The script continuously tracks the RSSI for a specified Bluetooth device and prints the median RSSI value and the estimated distance to the device every second.
"""


import os
import time
import numpy as np
from collections import deque

# Constants (to be calibrated in your environment)
RSSI_0 = -1  # RSSI at 1 meter (update this after calibration) Signal was so strong everywhere
N = 2.5  # Path loss exponent (update this after calibration) basically a weight for the signal strength
MOVING_AVERAGE_WINDOW = 10

def get_rssi_bluetooth(mac_address):

    try:
        output = os.popen(f"hcitool rssi {mac_address}").read()
        if "RSSI return value" in output:
            return int(output.split(":")[-1].strip())
    except ValueError:
        return None
    return None

def rssi_to_distance(rssi):
    try:
        distance = 10 ** ((RSSI_0 - rssi) / (10 * N))
        return round(distance, 2)
    except (ValueError, ZeroDivisionError):
        return None

if __name__ == "__main__":
    mac_address = "38:FC:98:78:D6:41"  #my personal laptop
    rssi_values = deque(maxlen=MOVING_AVERAGE_WINDOW)

    print(f"Tracking RSSI for device: {mac_address}")
    while True:
        rssi = get_rssi_bluetooth(mac_address)
        if rssi is not None:
            rssi_values.append(rssi)
            avg_rssi = np.median(rssi_values)  # Use median for smoothing
            distance = rssi_to_distance(avg_rssi)
            print(f"RSSI: {avg_rssi:.2f} dBm (Median), Estimated Distance: {distance} meters")
        else:
            print("Device not found or not connected.")
        
        time.sleep(1)
