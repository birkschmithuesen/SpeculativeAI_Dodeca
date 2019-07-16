"""
This program listens on OSC_IP_ADDRESS:OSC_PORT for incoming
sound vectors (array of floats) via OSC protocol address '/record_sound_vector'
and saves it to a buffer. After hitting ctrl-C in the terminal or receiving
any data on address '/stop' the recording process stops and the sound vectors
will be saved along neural net prediction output vectors based camera frames
that were saved during recording.
"""

import os
import threading
import signal
import sys
import csv
from pythonosc import osc_server, dispatcher
from conversation.vision_camera import Camera
from conversation import neuralnet, configuration, vision_camera

OSC_IP_ADDRESS = "localhost"
OSC_PORT = 8005

TRAININGS_SET_PATH = "./data/trainingsset_dodeca.csv"

SHOW_FRAMES = True #show window frames

CAMERA = Camera(224, 224)

trainingsset = []
trainingsset_final = []

stop_event = threading.Event()

def get_frame():
    """
    returns tuple with frame and name of file each in an array
    """
    for frames in CAMERA:
        cv2_img, pil_img = frames
        img_collection = [pil_img]
        names_of_file = ["test"]
        return img_collection, names_of_file, cv2_img

def process_trainingsset():
    """
    takes the trainings set images and transforms them to a
    512 dim vector based on the neural net and saves them together
    with the sound vector to the trainingsset_final list
    """
    MODEL = neuralnet.build_model()
    MODEL.summary()
    for set in trainingsset:
        soundvector = set [0]
        img_collection = set[1]
        names_of_file = set[2]
        cv2_img = set[3]
        if SHOW_FRAMES:
            vision_camera.cv2.imshow('frame', cv2_img)
            key = vision_camera.cv2.waitKey(20)
        activation_vectors, header, img_coll_bn = neuralnet.get_activations(\
            MODEL, img_collection, names_of_file)
        trainingsset_final.append((activation_vectors, soundvector))
    print("Finished processing trainings set")

def save_to_disk():
    """
    saves the trainings set from trainingsset_final to disk
    """
    with open(TRAININGS_SET_PATH, mode="w") as csv_file:
        fieldnames = ["image vector" + str (i) for i in range(512)]
        fieldnames.extend(["sound vector" + str (i) for i in range(5)])
        writer = csv.writer(csv_file, delimiter=" ")
        #writer.writerow(fieldnames)
        for image_vector, sound_vector in trainingsset_final:
            row = list(image_vector[0])
            row.extend(sound_vector)
            writer.writerow(row)
        abspath = os.path.realpath(csv_file.name)
        print("\n\nWritten trainings set to {}".format(abspath))

def record(address, *args):
    """
    Records incoming 5dim audio vector consisting of float values
    """
    soundvector = args
    img_collection, names_of_file, cv2_img = get_frame()
    trainingsset.append([soundvector, img_collection, names_of_file, cv2_img])

def osc_stop(address, *args):
    """
    Callback osc dispatcher to stop recording
    """
    stop_recording()

def signal_handler(signal, frame):
    """
    Callback for signal to stop recording
    """
    print('You pressed Ctrl+C!')
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

if __name__ == "__main__":
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/record_sound_vector", record)
    dispatcher.map("/stop", osc_stop)
    server = osc_server.ThreadingOSCUDPServer(
        (OSC_IP_ADDRESS, OSC_PORT), dispatcher)
    print("Serving on {}".format(server.server_address))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C to save current data to disk.')
    stop_event.wait()
    process_trainingsset()
    save_to_disk()
    sys.exit(0)
