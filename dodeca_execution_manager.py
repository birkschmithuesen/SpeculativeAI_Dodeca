"""
This script manages the execution and error handling of the sound synthesis
software SuperCollider in case the sound card is disconnected before
or after start of the host machine

Make sure, that the user running this script is allowed
to run sudo passwordless sudo commands for reboot and usbreset.
"""

import threading
import subprocess
from time import sleep
from queue import Queue

N_USB_DEVICE_CHECKS = 7

def usb_device_is_connected(device_string):
    """
    check if the given USB device is connected
    """
    usb_check = subprocess.run("lsusb | grep '{}'".format(device_string),\
            shell=True, stdout=subprocess.DEVNULL)
    if not usb_check.returncode == 0:
        print("USB device {} can't be detected".format(device_string))
        return False
    print("USB device {} detected".format(device_string))
    return True

def soundcard_is_connected():
    """
     check if the USB soundcard is connected
    """
    return usb_device_is_connected("C-Media Electronics, Inc.")

def camera_is_connected():
    """
    check if the USB camera is connected
    """
    return usb_device_is_connected("ARC International")

def devices_are_connected_multcheck(reset_function=None):
    """
    make sure devices are connected by checking N_USB_DEVICE_CHECKS times
    """
    for i in range(N_USB_DEVICE_CHECKS):
        if reset_function:
            reset_function()
        sleep(2)
        if soundcard_is_connected() and camera_is_connected():
            return True
    return False

def usb_reset_multcheck():
    """
    reset the USB ports
    """
    def reset_function():
        print("Resetting all USB ports")
        subprocess.run("echo /dev/bus/usb/*/* | xargs -t -n1 sudo usbreset",\
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return devices_are_connected_multcheck(reset_function)

def ensure_devices_are_connected():
    """
    check if devices are connected, if not reset the USB port, as a last resort
    restart the host system
    """
    if devices_are_connected_multcheck():
        return
    if usb_reset_multcheck():
        return
    # make sure that the user has rights to run passwordless sudo for reboot via `visudo`
    subprocess.Popen("sudo reboot".split())

def execute_supercollider():
    """
    run supercollider in background and return POpen object
    """
    cmd = "sclang /home/dodeca/Documents/SpeculativeAI_Dodeca/data/sound_synthesis/"\
            "SAI_soundSynthesis_04_masterEFX.scd"
    return subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def execute_dodeca_main():
    """
    run dodeca main in background and return POpen object
    """
    print("Starting dodeca main")
    cmd = "python3 /home/dodeca/Documents/SpeculativeAI_Dodeca/main.py"
    return subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def execute_supercollider_dodeca_main():
    """
    run both supercollider and dodeca main and block,
    if one fails return
    """
    print("Starting SuperCollider")
    dodeca_main = execute_dodeca_main()
    supercollider = execute_supercollider()
    def check_supercollider_stdout():
        for line in iter(supercollider.stderr.readline, b""):
            if b"JackAudioDriver::ProcessAsync: read error, stopping..." in line:
                print("JackAudioDriver stopped")
                break
            if b"Cannot initialize driver" in line:
                print("JackaudioDriver wasn't initialized")
                break
    def shutdown_execution():
        print("Shutting down execution of SuperCollider and dodeca main")
        dodeca_main.kill()
        supercollider.kill()
    supercollider_stdout_thread = threading.Thread(target=check_supercollider_stdout)
    supercollider_stdout_thread.start()
    while True:
        if dodeca_main.poll() is None and supercollider.poll() is None\
                and supercollider_stdout_thread.is_alive():
            continue
        shutdown_execution()
        return

def main():
    print("Starting Dodeca Manager")
    ensure_devices_are_connected()
    while True:
        execute_supercollider_dodeca_main()
        ensure_devices_are_connected()

if __name__ == "__main__":
    main()
