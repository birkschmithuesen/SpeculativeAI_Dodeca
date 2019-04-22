"""
Main file that executes the visual conversation part.
"""
import numpy as np
from pythonosc import udp_client
from conversation.vision_camera import Camera
from conversation import neuralnet, configuration, vision_camera

SLIDING_WINDOW_SIZE = 50 # number of frames in sliding window

TRANSFORM_USING_PCA = True  # True: transform from 512 to 5 using PCA. Otherwise, use random matrix

OSC_IP_ADDRESS = "localhost"
OSC_PORT = 8005
CONFIG_PATH = "./conversation_config"

SHOW_FRAMES = True #show window frames

CLIENT = udp_client.SimpleUDPClient(OSC_IP_ADDRESS, OSC_PORT)

CAMERA = Camera(224, 224)
#camera.show_capture()
MODEL = neuralnet.build_model()
MODEL.summary()
act_5dim_sliding = []
config_tracker = {}

config = configuration.ConversationConfig(CONFIG_PATH)
print(config.config)

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
        print("\033[93mClipped activations. Diff:")
        print(activation_diff)
        print("\033[0m")
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

while True:
    for frames in CAMERA:
        cv2_img, pil_img = frames
        if SHOW_FRAMES:
            vision_camera.cv2.imshow('frame', cv2_img)
            key = vision_camera.cv2.waitKey(20)
            process_key(key)
        img_collection = [pil_img]
        names_of_file = ["test"]
        break

    activation_vectors, header, img_coll_bn = neuralnet.get_activations(\
        MODEL, img_collection, names_of_file)

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
    activation_vector = clip_activation(activation_vector)

    CLIENT.send_message("/sound", activation_vector)
