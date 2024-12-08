"""
This script retrieves the MAC address of a connected Bluetooth device using the `bluetoothctl` command.
Functions:
    get_connected_device_mac(): Retrieves the MAC address of a connected Bluetooth device.
Usage:
    Run the script to print the MAC address of a connected Bluetooth device, if any.
"""


import os
import re

def get_connected_device_mac():
  
    try:

        output = os.popen("bluetoothctl devices Connected").read()
        match = re.search(r"([0-9A-F:]{17})", output)
        
        if match:
            return match.group(1)  # Return the MAC address
        else:
            return None
    except Exception as e:
        print(f"Error while retrieving MAC address: {e}")
        return None

if __name__ == "__main__":
    mac_address = get_connected_device_mac()
    if mac_address:
        print(f"Connected Device MAC Address: {mac_address}")
    else:
        print("No connected Bluetooth devices found.")



