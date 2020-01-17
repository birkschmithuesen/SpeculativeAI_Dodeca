"""
This script manages the execution and error handling of the sound synthesis
software SuperCollider in case the sound card or camera is disconnected before
or after start of the host machine or has any other issues that stops
the programs.

Make sure, that the user running this script is allowed
to run sudo passwordless sudo commands for reboot and usbreset.
"""

import threading
import subprocess
from time import time, sleep

N_USB_DEVICE_CHECKS = 3
WAIT_TIME_BEFORE_SUPERCOLLIDER_START_SECONDS = 0
EXECUTION_ATTEMPTS_RESET_TIMEOUT_SECONDS = 600

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
    sleep(WAIT_TIME_BEFORE_SUPERCOLLIDER_START_SECONDS)
    cmd = "sclang /home/dodeca/Documents/SpeculativeAI_Dodeca/data/sound_synthesis/"\
            "SAI_soundSynthesis_04_masterEFX.scd"
    return subprocess.Popen(cmd.split(), stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

def execute_dodeca_main():
    """
    run dodeca main in background and return POpen object
    """
    print("Starting dodeca main")
    cmd = "python3 /home/dodeca/Documents/SpeculativeAI_Dodeca/main.py"
    return subprocess.Popen(cmd.split(), stderr=subprocess.DEVNULL)

def execute_supercollider_dodeca_main():
    """
    run both supercollider and dodeca main and block,
    if one fails return
    """
    print("Starting SuperCollider")
    start_time = time()
    dodeca_main = execute_dodeca_main()
    supercollider = execute_supercollider()
    def check_supercollider_stdout():
        for line in iter(supercollider.stdout.readline, b""):
            if b"JackTemporaryException" in line:
                print("JackAudioDriver stopped")
                print(line)
                break
            if b"Cannot initialize driver" in line:
                print("JackAudioDriver couldn't be started")
                print(line)
                break
            if b"JackSocketClientChannel read fail" in line:
                print("JackAudioDriver couldn't be started")
                print(line)
                break
            if b"Server unresponsive:true" in line:
                print("JackAudioDriver stopped")
                print(line)
                break
    supercollider_stdout_thread = threading.Thread(target=check_supercollider_stdout)
    supercollider_stdout_thread.start()
    def shutdown_execution():
        print("Shutting down execution of SuperCollider and dodeca main")
        dodeca_main.kill()
        supercollider.kill()
        subprocess.run("killall jackd".split()) # necessary if we want to restart supercollider later
    while True:
        if dodeca_main.poll() is None and supercollider.poll() is None\
                and supercollider_stdout_thread.is_alive():
            continue
        print("One of the processes died.")
        print("Alive: dodeca_main: {} supercollider {} supercollider_stdout_thread: {}".format(dodeca_main.poll() is None, supercollider.poll() is None, supercollider_stdout_thread.is_alive()))
        shutdown_execution()
        return time() - start_time

def main():
    """
    first we check that the USB devices are visible, if yes we execute the software
    and monitor it's health. If one program dies, we check that the USB devices are visible
    and reset them if needed. If nothing helps we reboot the host.
    """
    print("Starting Dodeca Manager")
    execution_attempts = 0
    while True:
        execution_attempts += 1
        ensure_devices_are_connected()
        if execution_attempts >= N_USB_DEVICE_CHECKS:
            usb_reset_multcheck()
        if execution_attempts >= 2*N_USB_DEVICE_CHECKS:
            subprocess.Popen("sudo reboot".split())
        time_elapsed = execute_supercollider_dodeca_main()
        if time_elapsed > EXECUTION_ATTEMPTS_RESET_TIMEOUT_SECONDS:
            execution_attempts = 0

if __name__ == "__main__":
    main()
