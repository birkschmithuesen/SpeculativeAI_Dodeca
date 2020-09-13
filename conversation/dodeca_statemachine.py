import time
import numpy as np
from collections import deque
import random
import numpy as np
import cv2
import os
import screeninfo
from pythonosc import udp_client
from conversation.vision_camera import Camera
from conversation import neuralnet_vision, neuralnet_dictionary, configuration, vision_camera
from conversation import neuralnet_vision_inference
from conversation.neuralnet_vision_inference import InferenceModel
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.keras.preprocessing import image

LIVE_REPLAY = False  # replay the predictions live without buffer

SLIDING_WINDOW_SIZE = 50  # number of frames in sliding window

# True: transform from 512 to 5 using PCA. Otherwise, use neural net
TRANSFORM_USING_NEURAL_NET = True

OSC_IP_ADDRESS = "127.0.0.1"
OSC_PORT = 57120
CONFIG_PATH = "./data/conversation_config"
PCA_PATH = './data/pca.joblib'

SHOW_FRAMES = True  # show window frames

# these set tha random range for inserting/removing predictions
# N times into the prediction buffer
MESSAGE_RANDOMIZER_START = 0 # 0 - write the frame alays one time. 1 - write the message -1 till 2 times into the buffer
MESSAGE_RANDOMIZER_END = 2 #Experimenta: 1
SOUND_RANDOMIZER_START = 0 # Experimenta: -0.05 # set the minimum value, how much the volume of the different synths will be changed by chance
SOUND_RANDOMIZER_END = 0 # Experimenta: 0.05 # set the maximum value, how much the volume of the different synths will be changed by chance
SOUND_RANDOMIZER_MIN = 0 # Experimenta: -0.2
SOUND_RANDOMIZER_MAX = 0 # Experimenta: 0.2

# realfps * REPLAY_FPS_FACTOR is used for replaying the prediction buffer
MINIMUM_MESSAGE_LENGTH  = 4 # ignore all messages below this length
REPLAY_FPS_FACTOR = 2
PAUSE_LENGTH = 5 # length in frames of darkness that triggers pause event
# Threshhold defining pause if frame brightness is below the value
PAUSE_BRIGHTNESS_THRESH = 80 #this is the threshold for each pixel to be counted
PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH = 30 # this is the threshold for the number of counted pixels. Default is 50 for low ambient rooms
MAX_CONSTANT_STATE_DURATION_BEFORE_BRIGHTNESS_INCREASE_DECREASE = 30 # the number of seconds before the recording is stopped and the PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH increased
BRIGHTNESS_AVERAGES_BUFFER_MAXLEN = 15 # from how many replays do we calculate the brightness average for adjusting PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH
BRIGHTNESS_AUTO_ADJUST_FACTOR = 0.1 # the magnitude of adjustment to PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH after each replay based on brightness averages
PAUSE_BRIGHTNESS_DECREMENT = PAUSE_BRIGHTNESS_INCREMENT = 200 # constant change when stuck in a state other than replaying

PREDICTION_BUFFER_MAXLEN = 44 # 4 seconds * 11 fps

CLIENT = udp_client.SimpleUDPClient(OSC_IP_ADDRESS, OSC_PORT)

ZOOM_AREA_WIDTH = 380 #480 is full sensor width
ZOOME_AREA_HEIGHT = 380 #480 is full sensor width

FINAL_IMAGE_WIDTH = FINAL_IMAGE_HEIGHT = 224

CAMERA = Camera(FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT, ZOOM_AREA_WIDTH, ZOOME_AREA_HEIGHT)

if type(tf.contrib) != type(tf): tf.contrib._warning = None

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var wildictionary_moprediction_buffer_remove_pausedel.h5l prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph, graph_def


if os.path.isfile(neuralnet_vision_inference.TENSORRT_MODEL_PATH):
    print("Using optimized inference. {} found!".format(neuralnet_vision_inference.TENSORRT_MODEL_PATH))
    MODEL = InferenceModel()
    MODEL_GRAPH = None
else:
    print("Using unoptimized inference. {} not found!".format(neuralnet_vision_inference.TENSORRT_MODEL_PATH))
    MODEL = neuralnet_vision
    MODEL_GRAPH = neuralnet_vision.build_model()
    MODEL_GRAPH.summary()


class FPSCounter():
    """
    This class tracks average fps
    """

    def __init__(self):
        self.last_timestamp = False
        self.end_timestamp = False
    def record_end_new_frame(self, n_frames):
        if self.last_timestamp and not self.end_timestamp:
            time_now = time.time()
            time_delta = time_now - self.last_timestamp
            self.fps_sum = time_delta
            self.n_frames = n_frames

    def record_start_new_frame(self):
        self.last_timestamp = time.time()
        self.end_timestamp = False

    def get_average_fps(self):
        return self.n_frames / self.fps_sum


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
    Return 5 dim vector for neural net and
    for pca the same inside a list object
    """
    if TRANSFORM_USING_NEURAL_NET:
        prediction_input = [activations]
        prediction_input = np.asarray(prediction_input)
        prediction_input.shape = (1, neuralnet_dictionary.INPUT_DIM)
        res_5dim = neuralnet_dictionary.model.predict(prediction_input)[0]
    else:
        pca = neuralnet_vision.load(PCA_PATH)
        res_5dim = pca.transform(activations)
        # if you want to add some gaussian noise to the 5dim vectors you can uncomment
        # the following 2 lines
        # MG = np.random.normal(0, scale=0.1, size=act_5dim.shape)
        # act_5dim = act_5dim + MG
    return res_5dim


def count_image_bright_pixels(image_frame):
    image = np.zeros((224, 224, 3), np.uint8)
    cv2.cvtColor(image_frame, cv2.COLOR_RGB2HSV, image)
    brightnessvalues = image[:, :, 2]
    counter = np.sum(brightnessvalues > PAUSE_BRIGHTNESS_THRESH)
    print("Pixels above threshold: {}\n".format(counter))
    return counter

def contains_darkness(image_frame):
    """
    Return true if average frame brightness is
    below PAUSE_BRIGHTNESS_THRESH
    """
    counter = count_image_bright_pixels(image_frame)
    print( "PAUSE_BRIGHTNESS_MIN_NUM_PIXELS = {}".format(PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH))
    return counter < PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH


def contains_darkness_pause_detected(image_frame):
    """
    returns tuple first one being True if a pause in the fft stream
    was detected else return False. The second one is True if the
    frame contains silence (sound below threshold) or False otherwise.
    """
    global pause_counter
    is_dark = contains_darkness(image_frame)
    if is_dark:
        print("Pause Counter: {}\n".format(pause_counter))
        if pause_counter >= PAUSE_LENGTH:
            pause_counter = 0
            return is_dark, True
        pause_counter += 1
    else:
        pause_counter = 0
    return is_dark, False


def prediction_buffer_remove_pause():
    """
    Removes dark pause frames at the end of
    prediction_buffer
    """
    global prediction_counter
    # -1 prediction_buffer_remove_pausebecause the last pause frame wrecordon't be recorded in state machine
    last_frame_counter = prediction_counter - (PAUSE_LENGTH - 1)
    if len(prediction_buffer) == 0:
        return
    while(prediction_buffer[-1][1] > last_frame_counter):
        prediction_buffer.pop()
        if len(prediction_buffer) == 0:
           return

def play_buffer():
    """
    Send out all sframes_to_reound predictions in the buffer with the
    configured REPLAY_FPS_FACTOR until it's empty
    """
    global brightness_averages
    brightness_values = []
    if not LIVE_REPLAY:
        real_fps = fpscounter.get_average_fps()
        replay_fps = real_fps * REPLAY_FPS_FACTOR
    else:
        replay_fps = 0
    print("Playing Buffer with {} FPS\n".format(replay_fps))
    while len(prediction_buffer) > 0:
        img_collection, names, cv2_img = get_frame()
        brightness_value = count_image_bright_pixels(cv2_img)
        brightness_values.append(brightness_value)
        prediction = prediction_buffer.popleft()[0]
        print(prediction)
        CLIENT.send_message("/sound", prediction)
        if replay_fps > 0:
            # ensure playback speed matches framerate
            time.sleep(1 / replay_fps)
    avg_brightness = sum(brightness_values)/len(brightness_values)
    brightness_averages.append(avg_brightness)
    adjust_brightness_threshhold()

def check_brightness_threshhold():
    """
    Check that the PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH is within
    a valid range and fix it if not
    """
    global PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH
    if PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH < 0:
        PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH = 0
        print("Clipped PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH to MIN")
    elif PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH > FINAL_IMAGE_HEIGHT * FINAL_IMAGE_WIDTH:
        PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH = FINAL_IMAGE_HEIGHT * FINAL_IMAGE_WIDTH
        print("Clipped PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH to MAX")
    PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH = int(PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH)

def adjust_brightness_threshhold():
    """
    Adjust PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH for better
    activity detection
    """
    global brightness_averages
    global PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH
    brightness_sum = sum(brightness_averages)
    brightness_avg = brightness_sum / len(brightness_averages)

    diff = PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH - brightness_avg
    PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH -= BRIGHTNESS_AUTO_ADJUST_FACTOR * diff
    check_brightness_threshhold()


def get_frame():
    """
    returns tuple with frame andwar name of file each in an array
    """
    for frames in CAMERA:
        cv2_img, pil_img = frames
        if SHOW_FRAMES:
            vision_camera.cv2.imshow("dodeca", cv2_img)
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

    if TRANSFORM_USING_NEURAL_NET:
        return np.array(act_5dim, dtype="float64")
    act_5dim_sliding.append(act_5dim[0])
    if len(act_5dim_sliding) > SLIDING_WINDOW_SIZE:
        act_5dim_sliding.pop(0)

    # activations = neuralnet_vision.sigmoid(act_5dim, coef=0.05)  # Sigmoid function
    #activations = act_5dim
    mins = np.array(config.config["mins"], dtype="float64")
    maxs = np.array(config.config["maxs"], dtype="float64")
    mins_track = neuralnet_vision.np.min(act_5dim_sliding, 0)
    maxs_track = neuralnet_vision.np.max(act_5dim_sliding, 0)
    config_tracker["mins"] = mins_track
    config_tracker["maxs"] = maxs_track
    activations_5dim = (act_5dim_sliding - mins) / (maxs - mins)
    activation_vector = activations_5dim[-1]
    return clip_activation(activation_vector)

def soundvector_postprocessing(prediction_vector):
    """
    adds some random noise or any other function to the sound vector,
    to add purpose to the answer
    """
    for i in range (0, len(prediction_vector)):
        soundvector_purpose[i] = np.clip(soundvector_purpose[i] + random.uniform(SOUND_RANDOMIZER_START, SOUND_RANDOMIZER_END), SOUND_RANDOMIZER_MIN, SOUND_RANDOMIZER_MAX)
    #print(soundvector_purpose)
    prediction_vector = np.clip(prediction_vector + soundvector_purpose, 0, 1)
    #prediction_vector[0] = np.clip(prediction_vector[0] + random.uniform(VOLUME_RANDOMIZER_START, VOLUME_RANDOMIZER_END), 0, 1)
    #prediction_vector[6] = np.clip(prediction_vector[6] + random.uniform(VOLUME_RANDOMIZER_START, VOLUME_RANDOMIZER_END), 0, 1)
    return prediction_vector

class State:
    def run(self):
        assert 0, "rfpscounter.record_start_new_frameun not implemented"

    def next(self, input):
        assert 0, "next not implemented"


class StateMachine:
    def __init__(self, initialState):
        self.currentState = initialState
        self.currentState.run()

    def run(self, input=None):
        img_collection, names_of_file, cv2_img = get_frame()
        self.currentState = self.currentState.next(cv2_img)
        self.currentState.run((img_collection, names_of_file))

class Waiting(State):
    """
    Waiting for a non dark frame to transition
    to recording state
    """

    def run(self, image_frames=None):
        pass

    def next(self, image_frame):
        global waiting_start_time
        global PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH

        frame_contains_darkness, _pause_detected = contains_darkness_pause_detected(
            image_frame)

        if not waiting_start_time:
            waiting_start_time = time.time()
        if time.time() - waiting_start_time > MAX_CONSTANT_STATE_DURATION_BEFORE_BRIGHTNESS_INCREASE_DECREASE:
            PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH -= PAUSE_BRIGHTNESS_DECREMENT
            print("Recording above max duration. Adjusting PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH to {}".format(PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH))
            check_brightness_threshhold()

        if frame_contains_darkness:
            return DodecaStateMachine.waiting
        print("Transitioned: Recording")
        fpscounter.record_start_new_frame()
        waiting_start_time = None
        return DodecaStateMachine.recording

class Recording(State):
    """
    Recording the image prediction frames and waiting for detecting a pause
    to transition to replay statec
    """

    def run(self, image_frames):
        global prediction_counter
        global frames_to_remove
        global should_increase_length

        img_collection, names_of_file = image_frames
        activation_vectors, header, img_coll_bn = MODEL.get_activations(
            MODEL_GRAPH, img_collection, names_of_file)
        activation_vector = prediction_postprocessing(activation_vectors)
        activation_vector = soundvector_postprocessing(activation_vector)
        prediction_counter += 1
        if LIVE_REPLAY:
            random_value = 0
        else:
            random_value = random.randint(
                MESSAGE_RANDOMIZER_START, MESSAGE_RANDOMIZER_END)
            should_increase_length = should_increase_length + random.uniform(-1, 1)
            should_increase_length = np.clip(should_increase_length, -5, 5)
        #print("should increase length", should_increase_length)
        prediction_buffer.append((activation_vector, prediction_counter))
        if should_increase_length>0:
            for i in range(random_value):
                prediction_buffer.append((activation_vector, prediction_counter))
        else:
            frames_to_remove += random_value
        while(frames_to_remove > 0):
                 if len(prediction_buffer) > MINIMUM_MESSAGE_LENGTH:
                     prediction_buffer.pop()
                     frames_to_remove -= 1
                 else:
                     break

    def next(self, image_frame):
        global prediction_counter
        global recording_start_time
        global PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH

        _frame_contains_darkness, pause_detected = contains_darkness_pause_detected(
            image_frame)

        if not recording_start_time:
            recording_start_time = time.time()
        if time.time() - recording_start_time > MAX_CONSTANT_STATE_DURATION_BEFORE_BRIGHTNESS_INCREASE_DECREASE:
            PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH += PAUSE_BRIGHTNESS_INCREMENT
            check_brightness_threshhold()
            print("Recording above max duration. Adjusting PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH to {}".format(PAUSE_BRIGHTNESS_MIN_NUM_PIXELS_ABOVE_THRESH))
            pause_detected = True

        #print("Prediction Counter: ", prediction_counter)
        #print("len(prediction_buffer): ", len(prediction_buffer))
        if pause_detected:
            recording_start_time = None
            prediction_buffer_remove_pause()
            print("Prediction Counter: ")
            print(prediction_counter)
            print("len(prediction_buffer): ")
            print(len(prediction_buffer))
            fpscounter.record_end_new_frame(prediction_counter)
            if len(prediction_buffer) < MINIMUM_MESSAGE_LENGTH:
                 print("Transitioned: Waiting")
                 prediction_buffer.clear()
                 prediction_counter = frames_to_remove = 0
                 return DodecaStateMachine.waiting
            print("Transitioned: Replaying")
            prediction_counter = frames_to_remove = 0
            return DodecaStateMachine.replaying
        else:
            if len(prediction_buffer) == PREDICTION_BUFFER_MAXLEN:
                fpscounter.record_end_new_frame(PREDICTION_BUFFER_MAXLEN)
            return DodecaStateMachine.recording


class Replaying(State):
    """
    Replaying the recorded image based predictions and after
    finishing transitioning to waiting state.
    """

    def run(self, image_frames=None):
        play_buffer()

    def next(self, image_frame):
        return DodecaStateMachine.waiting


class DodecaStateMachine(StateMachine):
    def __init__(self):
        StateMachine.__init__(self, DodecaStateMachine.waiting)


act_5dim_sliding = []
config_tracker = {}

if TRANSFORM_USING_NEURAL_NET:
    neuralnet_dictionary.run()

config = configuration.ConversationConfig(CONFIG_PATH)
print(config.config)

prediction_buffer = deque(maxlen=PREDICTION_BUFFER_MAXLEN)
pause_counter = 0
prediction_counter = 0
frames_to_remove = 0
fpscounter = FPSCounter()
should_increase_length = 0
soundvector_purpose = np.zeros(shape=(8))
recording_start_time = None
waiting_start_time = None
brightness_averages = deque(maxlen=BRIGHTNESS_AVERAGES_BUFFER_MAXLEN)

DodecaStateMachine.waiting = Waiting()
DodecaStateMachine.recording = Recording()
DodecaStateMachine.replaying = Replaying()

screen = screeninfo.get_monitors().pop()

vision_camera.cv2.namedWindow("dodeca", cv2.WINDOW_NORMAL)
vision_camera.cv2.resizeWindow("dodeca", (int(screen.height-52), int(screen.height)))
vision_camera.cv2.moveWindow("dodeca", int(screen.width-screen.height+52), 0)
#vision_camera.cv2.setWindowProperty("dodeca",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


if LIVE_REPLAY:
    def new_next_recording(image_frame):
        prediction_counter = 0
        return DodecaStateMachine.replaying

    def new_next_replaying(image_frame):
        return DodecaStateMachine.recording
    DodecaStateMachine.recording.next = new_next_recording
    DodecaStateMachine.replaying.next = new_next_replaying
