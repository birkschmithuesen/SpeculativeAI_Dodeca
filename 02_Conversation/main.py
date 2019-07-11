"""
Main file that executes the visual conversation part.
"""
import time
import numpy as np
from collections import deque
import random
import numpy as np
import cv2
from pythonosc import udp_client
from conversation.vision_camera import Camera
from conversation import neuralnet, configuration, vision_camera

SLIDING_WINDOW_SIZE = 50 # number of frames in sliding window

TRANSFORM_USING_PCA = True  # True: transform from 512 to 5 using PCA. Otherwise, use random matrix

OSC_IP_ADDRESS = "2.0.0.2"
OSC_PORT = 57120
CONFIG_PATH = "./conversation_config"

SHOW_FRAMES = True #show window frames

# these set tha random range for inserting a predictions
# multiple times (inluding 0, if set to start at 0)
# into the prediction buffer
MESSAGE_RANDOMIZER_START = 0
MESSAGE_RANDOMIZER_END = 4


FPS = 15 # fps used for replaying the prediction buffer
PAUSE_LENGTH = 25 # length in frames of darkness that triggers pause event
PAUSE_BRIGHTNESS_THRESH = 84 # Threshhold defining pause if frame brightness is below the value
PREDICTION_BUFFER_MAXLEN = 200 # 10 seconds * 44.1 fps

CLIENT = udp_client.SimpleUDPClient(OSC_IP_ADDRESS, OSC_PORT)

CAMERA = Camera(224, 224)
#camera.show_capture()
MODEL = neuralnet.build_model()
MODEL.summary()
act_5dim_sliding = []
config_tracker = {}

config = configuration.ConversationConfig(CONFIG_PATH)
print(config.config)

prediction_buffer = deque(maxlen=PREDICTION_BUFFER_MAXLEN)
pause_counter = 0

def save_current_config():
    """
    safe all current settings of the app
    """
    print("Saving current config")
    config.save_config(config_tracker)

def clip_activation(activation):
    """
    Limit activation values betwenn 0 and 1
    """
    act_new = []
    for act in activation:
        val = act
        if act < 0.0:
            val = 0.0
        if act > 1.0:
            val = 1.0
        act_new.append(val)
    act_new_np = np.array(act_new, dtype="float64")
    activation_diff = activation - act_new_np
    if np.count_nonzero(activation_diff) > 0:
        print("Clipped activations. Diff:")
        print(activation_diff)
        print("")
    return act_new_np

def process_key(key_input):
    """
    quit programm on 'q' and save current config on 's'
    """
    if key_input & 0xFF == ord('q'):
        vision_camera.cv2.destroyAllWindows()
        CAMERA.release()
        exit(0)
    elif key_input & 0xFF == ord('s'):
        save_current_config()

def reduce_to_5dim(activations):
    """
    reduce 512 dim vector to 5 dim vector
    """
    if TRANSFORM_USING_PCA:
        pca = neuralnet.load('pca.joblib')
        res_5dim = pca.transform(activations)
        # if you want to add some gaussian noise to the 5dim vectors you can uncomment
        # the following 2 lines
        # MG = np.random.normal(0, scale=0.1, size=act_5dim.shape)
        # act_5dim = act_5dim + MG
    else:
        M, MG = neuralnet.get_M_and_MG_from_file()
        res_5dim = activations @ M  # matrix multiplication
    return res_5dim

def is_pause(frame):
    """
    return True if a pause in the image stream was detected else return False
    """
    global pause_counter
    image = np.zeros((224,224,3), np.uint8)
    cv2.cvtColor(frame, cv2.COLOR_RGB2HSV, image)
    brightness = np.mean(image[:,:,2])
    print("Brightness: {}".format(brightness))
    print("")
    if brightness < PAUSE_BRIGHTNESS_THRESH:
        print("Pause Counter: ", pause_counter)
        print("")
        if pause_counter >= PAUSE_LENGTH:
            pause_counter = 0
            return True
        pause_counter += 1
    else:
        pause_counter = 0
    return False

def play_buffer():
    """
    Send out all sound predictions in the buffer with the
    configured FPS until it's empty
    """
    while len(prediction_buffer) > 0:
        print("Playing Buffer ")
        CLIENT.send_message("/sound", prediction_buffer.popleft())
        time.sleep(1/FPS) #ensure playback speed matches framerate

def get_frame():
    """
    returns tuple with frame and name of file each in an array
    """
    for frames in CAMERA:
        cv2_img, pil_img = frames
        if SHOW_FRAMES:
            vision_camera.cv2.imshow('frame', cv2_img)
            key = vision_camera.cv2.waitKey(20)
            process_key(key)
        img_collection = [pil_img]
        names_of_file = ["test"]
        return img_collection, names_of_file, cv2_img

def prediction_postprocessing(activation_vectors):
    """
    Reduces vector dimension from 512 to 5 and then normalizes clips the
    resulting vector
    """
    act_5dim = reduce_to_5dim(activation_vectors)

    act_5dim_sliding.append(act_5dim[0])
    if len(act_5dim_sliding) > SLIDING_WINDOW_SIZE:
        act_5dim_sliding.pop(0)

    #activations = neuralnet.sigmoid(act_5dim, coef=0.05)  # Sigmoid function
    #activations = act_5dim
    mins = np.array(config.config["mins"], dtype="float64")
    maxs = np.array(config.config["maxs"], dtype="float64")
    mins_track = neuralnet.np.min(act_5dim_sliding, 0)
    maxs_track = neuralnet.np.max(act_5dim_sliding, 0)
    config_tracker["mins"] = mins_track
    config_tracker["maxs"] = maxs_track
    activations_5dim = (act_5dim_sliding - mins) / (maxs - mins)
    activation_vector = activations_5dim[-1]
    return clip_activation(activation_vector)


while True:
    img_collection, names_of_file, cv2_img = get_frame()

    if(is_pause(cv2_img)):
        play_buffer()
        continue

    activation_vectors, header, img_coll_bn = neuralnet.get_activations(\
        MODEL, img_collection, names_of_file)

    activation_vector = prediction_postprocessing(activation_vectors)

    random_value = random.randint(MESSAGE_RANDOMIZER_START, MESSAGE_RANDOMIZER_END)
    for i in range(random_value):
        prediction_buffer.append(activation_vector)
