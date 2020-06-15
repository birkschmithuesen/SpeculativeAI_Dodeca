"""
This program listens on OSC_IP_ADDRESS:OSC_PORT for incoming
sound vectors (array of floats) via OSC protocol address '/record_sound_vector'
and saves it to a buffer. After hitting ctrl-C in the terminal or receiving
any data on address '/stop' the recording process stops and the sound vectors
will be saved along neural net prediction output vectors based camera frames
that were saved during recording.

HowTo:
- Check IP address in Vezer and SuperCollider
- Vezer(Trainings Notebook) sends sound data to SuperCollider
- Vezer sends '/sendTrainData' command to SuperCollider. By this command,
SuperCollider will send all sound messages stored in one vector
'/record_sound_vector' to this Python Dispatcher
- send '/stop' message from Vezer to this patch, to stop recording
"""

import os
import threading
import signal
import sys
import platform
import csv
from pythonosc import osc_server, dispatcher
from conversation.vision_camera import Camera
from conversation import neuralnet_vision_inference, configuration, vision_camera

OSC_IP_ADDRESS = "0.0.0.0"
OSC_PORT = 8005

TRAININGS_SET_PATH = "./data/trainingsset_dodeca.csv"

SHOW_FRAMES = True  # show window frames

ZOOM_AREA_WIDTH = 380
ZOOME_AREA_HEIGHT = 380

CAMERA = Camera(224, 224, ZOOM_AREA_WIDTH, ZOOME_AREA_HEIGHT)

MODEL = neuralnet_vision_inference.InferenceModel()

trainingsset = []
trainingsset_final = []

stop_event = threading.Event()

def get_frame():
    """
    returns tuple with frame andwar name of file each in an array
    """
    for frames in CAMERA:
        cv2_img, pil_img = frames
        if SHOW_FRAMES:
            vision_camera.cv2.imshow('frame', cv2_img)
            key = vision_camera.cv2.waitKey(20)
        img_collection = [pil_img]
        names_of_file = ["test"]
        return img_collection, names_of_file, cv2_img


def process_trainingsset():
    """
    takes the trainings set images and transforms them to a
    512 dim vector based on the neural net and saves them together
    with the sound vector to the trainingsset_final list
    """
   # ->moved to line 31 MODEL = neuralnet_vision_inference.InferenceModel()
    for set in trainingsset:
        soundvector = set[0]
        img_collection = set[1]
        names_of_file = set[2]
        cv2_img = set[3]
        activation_vectors, header, img_coll_bn = MODEL.get_activations(
            MODEL, img_collection, names_of_file)
        trainingsset_final.append((activation_vectors, soundvector))
    print("Finished processing trainings set")


def save_to_disk():
    """
    saves the trainings set from trainingsset_final to disk
    """
    if len(trainingsset_final) == 0:
        print("No trainings data received. Nothing written to disk.\n")
        return
    with open(TRAININGS_SET_PATH, mode="w") as csv_file:
        fieldnames = ["image vector" + str(i) for i in range(512)]
        fieldnames.extend(["sound vector" + str(i) for i in range(5)])
        writer = csv.writer(csv_file, delimiter=" ")
        # writer.writerow(fieldnames)
        for image_vector, sound_vector in trainingsset_final:
            row = list(image_vector[0])
            row.extend(sound_vector)
            writer.writerow(row)
        abspath = os.path.realpath(csv_file.name)
        print("\n\nWritten trainings set to {}".format(abspath))


def record(address, *args):
    """ blocking
    Records incoming 5dim audio vector consisting of float values
    """
    soundvector = args
    img_collection, names_of_file, cv2_img = get_frame()
    trainingsset.append([soundvector, img_collection, names_of_file, cv2_img])


def osc_stop(address, *args):
    """
    Callback osc dispatcher to stop recording
    """
    print("received /stop")
    stop_recording()


def stop_recording():
    """
    Stops the recording and processes the already recorded frames
    and saves the result to disk
    """
    def stop():
        server.shutdown()
        server.server_close()
        stop_event.set()
    threading.Thread(target=stop, daemon=True).start()


def start_recording():
    """
    Execute the trainingsrecorder
    """
    global server
    dispatcher_server = dispatcher.Dispatcher()
    dispatcher_server.map("/record_sound_vector", record)
    dispatcher_server.map("/stop", osc_stop)
    server = osc_server.BlockingOSCUDPServer(
        (OSC_IP_ADDRESS, OSC_PORT), dispatcher_server)
    print("Serving on {}".format(server.server_address))
    #threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    server = start_recording()
    server.serve_forever()
    process_trainingsset()
    save_to_disk()
    sys.exit(0)
