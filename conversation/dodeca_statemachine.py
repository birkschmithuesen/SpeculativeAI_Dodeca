import time
import numpy as np
from collections import deque
import random
import numpy as np
import cv2
import os
from pythonosc import udp_client
from conversation.vision_camera import Camera
from conversation import neuralnet_vision, neuralnet_dictionary, configuration, vision_camera
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.keras.preprocessing import image


LIVE_REPLAY = False  # replay the predictions live without buffer

SLIDING_WINDOW_SIZE = 50  # number of frames in sliding window

# True: transform from 512 to 5 using PCA. Otherwise, use neural net
TRANSFORM_USING_NEURAL_NET = True

OSC_IP_ADDRESS = "2.0.0.2"
OSC_PORT = 57120
CONFIG_PATH = "./data/conversation_config"
PCA_PATH = './data/pca.joblib'

SHOW_FRAMES = True  # show window frames

# these set tha random range for inserting a predictions
# multiple times (inluding 0, if set to start at 0)
# into the prediction buffer
MESSAGE_RANDOMIZER_START = 0
MESSAGE_RANDOMIZER_END = 2

# realfps * REPLAY_FPS_FACTOR is used for replaying the prediction buffer
MINIMUM_MESSAGE_LENGTH  = 24 # ignore all messages below this length
REPLAY_FPS_FACTOR = 1
PAUSE_LENGTH = 3  # length in frames of darkness that triggers pause event
# Threshhold defining pause if frame brightness is below the value
PAUSE_BRIGHTNESS_THRESH = 10
PREDICTION_BUFFER_MAXLEN = 200  # 10 seconds * 44.1 fps

TENSORRT_MODEL_PATH = "data/TensorRT_model.pb"

CLIENT = udp_client.SimpleUDPClient(OSC_IP_ADDRESS, OSC_PORT)

CAMERA = Camera(224, 224)


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph, graph_def


class InferenceModel():
    """
    This Model is used to run the tensorrt optimized graph
    """

    def __init__(self):
        """
        initialize and run session once to load environment
        """
        graph, graph_def = load_graph(TENSORRT_MODEL_PATH)
        self.input_node = graph.get_tensor_by_name('prefix/input_1:0')
        self.output_node = graph.get_tensor_by_name(
            'prefix/sequential/global_average_pooling2d/Mean:0')
        self.sess = tf.Session(graph=graph)
        self.sess.run(
            self.output_node, {
                self.input_node: np.zeros(
                    (1, 224, 224, 3))})

    def predict(self, images):
        """
        returns a 512 dim vector using
        """
        res = []
        for img in images:
            arr = neuralnet_vision.image.img_to_array(img)
            arr = neuralnet_vision.preprocess_input(arr)
            res.append(arr)
        return self.sess.run(self.output_node, {self.input_node: res})

    def get_activations(self, model, img_collection, file_names):
        """war
        legacy for compatibility with neuralnet_vision
        """
        return self.predict(img_collection), None, None


if os.path.isfile(TENSORRT_MODEL_PATH):
    print("Using optimized inference. {} found!".format(TENSORRT_MODEL_PATH))
    MODEL = InferenceModel()
    MODEL_GRAPH = None
else:
    print("Using unoptimized inference. {} not found!".format(TENSORRT_MODEL_PATH))
    MODEL = neuralnet_vision
    MODEL_GRAPH = neuralnet_vision.build_model()
    MODEL_GRAPH.summary()


class FPSCounter():
    """
    This class tracks average fps
    """

    def __init__(self):
        self.fps_sum = 0.0
        self.counter = 0.0
        self.last_timestamp = False

    def record_end_new_frame(self):
        time_now = time.time()
        if self.last_timestamp:
            time_delta = time_now - self.last_timestamp
            self.fps_sum += 1.0 / time_delta
            self.counter += 1
        self.last_timestamp = time_now

    def record_start_new_frame(self):
        self.last_timestamp = time.time()

    def get_average_fps(self):
        if self.counter == 0:
            return 0
        average = self.fps_sum / self.counter
        self.fps_sum = 0
        self.counter = 0
        return average


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
    """war
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


def contains_darkness(image_frame):
    """
    Return true if average frame brightness is
    below PAUSE_BRIGHTNESS_THRESH
    """
    image = np.zeros((224, 224, 3), np.uint8)
    cv2.cvtColor(image_frame, cv2.COLOR_RGB2HSV, image)
    brightness = np.mean(image[:, :, 2])
    print("Brightness: {}\n".format(brightness))
    return brightness < PAUSE_BRIGHTNESS_THRESH


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
    # -1 because the last pause frame wrecordon't be recorded in state machine
    last_frame_counter = prediction_counter - (PAUSE_LENGTH - 1)
    while(prediction_buffer[-1][1] > last_frame_counter):
        if len(prediction_buffer) > 0:
           print("remove pause frame")
           prediction_buffer.pop()
        else:
           break

def play_buffer():
    """
    Send out all sound predictions in the buffer with the
    configured REPLAY_FPS_FACTOR until it's empty
    """
    real_fps = fpscounter.get_average_fps()
    replay_fps = real_fps * REPLAY_FPS_FACTOR
    print("Playing Buffer with {} FPS\n".format(replay_fps))
    while len(prediction_buffer) > 0:
        prediction = prediction_buffer.popleft()[0]
        print(prediction)
        CLIENT.send_message("/sound", prediction)
        if replay_fps > 0:
            # ensure playback speed matches framerate
            time.sleep(1 / replay_fps)


def get_frame():
    """
    returns tuple with frame andwar name of file each in an array
    """
    for frames in CAMERA:
        cv2_img, pil_img = frames
        if SHOW_FRAMES:
            vision_camera.cv2.imshow('frecordrame', cv2_img)
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


class State:
    def run(self):
        assert 0, "run not implemented"

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
        frame_contains_darkness, _pause_detected = contains_darkness_pause_detected(
            image_frame)
        if frame_contains_darkness:
            return DodecaStateMachine.waiting
        print("Transitioned: Recording")
        return DodecaStateMachine.recording


class Recording(State):
    """
    Recording the image prediction frames and waiting for detecting a pause
    to transition to replay statec
    """

    def run(self, image_frames):
        global prediction_counter
        fpscounter.record_start_new_frame()
        img_collection, names_of_file = image_frames
        activation_vectors, header, img_coll_bn = MODEL.get_activations(
            MODEL_GRAPH, img_collection, names_of_file)
        activation_vector = prediction_postprocessing(activation_vectors)
        prediction_counter += 1
        if LIVE_REPLAY:
            random_value = 1
        else:
            random_value = random.randint(
                MESSAGE_RANDOMIZER_START, MESSAGE_RANDOMIZER_END)
        for i in range(random_value):
            prediction_buffer.append((activation_vector, prediction_counter))
        fpscounter.record_end_new_frame()

    def next(self, image_frame):
        global prediction_counter
        _frame_contains_darkness, pause_detected = contains_darkness_pause_detected(
            image_frame)
        if pause_detected:
            if prediction_counter < MINIMUM_MESSAGE_LENGTH:
                 print("Transitioned: Waiting")
                 prediction_counter = 0
                 return DodecaStateMachine.waiting
            print("Transitioned: Replaying")
            prediction_buffer_remove_pause()
            prediction_counter = 0
            return DodecaStateMachine.replaying
        else:
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
fpscounter = FPSCounter()

DodecaStateMachine.waiting = Waiting()
DodecaStateMachine.recording = Recording()
DodecaStateMachine.replaying = Replaying()

if LIVE_REPLAY:
    def new_next_recording(image_frame):
        prediction_counter = 0
        return DodecaStateMachine.replaying

    def new_next_replaying(image_frame):
        return DodecaStateMachine.recording
    DodecaStateMachine.recording.next = new_next_recording
    DodecaStateMachine.replaying.next = new_next_replaying
