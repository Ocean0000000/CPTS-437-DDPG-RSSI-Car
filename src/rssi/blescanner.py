"""
This script scans for Bluetooth Low Energy (BLE) devices and checks if any of them advertise a specific Service UUID.
It uses the Bleak library to perform the scanning asynchronously.
Constants:
    TARGET_UUID (str): The UUID of the target BLE service to search for.
Functions:
    scan_for_uuid(): Asynchronously scans for BLE devices and prints out whether any device advertises the target UUID.
Usage:
    The script runs the scan_for_uuid function using asyncio to perform the BLE scan.
"""




import asyncio
from bleak import BleakScanner

TARGET_UUID = "6E9E10D6-5184-4939-BE4A-F8330B816681"  # Trying to use airpods to test this

async def scan_for_uuid():
    print(f"Scanning for devices with Service UUID: {TARGET_UUID}")
    devices = await BleakScanner.discover()
    
    for device in devices:
        # Check advertised services for the target UUID
        if TARGET_UUID in (device.metadata.get("uuids") or []):
            print(f"Found device with matching UUID: {device.name} ({device.address})")
        else:
            print(f"Other device: {device.name} ({device.address})")

# Run the scanning function
asyncio.run(scan_for_uuid())
