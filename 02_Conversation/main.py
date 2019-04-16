"""
Main file that executes the visual conversation part.
"""
import time
from pythonosc import udp_client
from conversation.camera import Camera
from conversation import neuralnet

TRANSFORM_FROM_CNN_DIM_TO_SOUND_DIM = True
TRANSFORM_USING_PCA = True  # True: transform form 512 to 5 using PCA. Otherwise, using random matrix

OSC_IP_ADDRESS = "localhost"
OSC_PORT = 8005

client = udp_client.SimpleUDPClient(OSC_IP_ADDRESS, OSC_PORT)

camera = Camera(224, 224)
#camera.show_capture()
model = neuralnet.build_model()
model.summary()
count = 0
for frames in camera:
    cv2_img, pil_img = frames

    img_collection = [pil_img]
    names_of_file = ["test"]

    activations, header, img_coll_bn = neuralnet.get_activations(model, img_collection, names_of_file)

    if TRANSFORM_FROM_CNN_DIM_TO_SOUND_DIM:
        if TRANSFORM_USING_PCA:
            pca = neuralnet.load('pca.joblib')
            act_5dim = pca.transform(activations)
            # if you want to add some gaussian noise to the 5dim vectors you can uncomment the following 2 lines
            # MG = np.random.normal(0, scale=0.1, size=act_5dim.shape)
            # act_5dim = act_5dim + MG
        else:
            M, MG = neuralnet.get_M_and_MG_from_file()
            act_5dim = activations @ M  # matrix multiplication

        print(act_5dim)

        activations = neuralnet.sigmoid(act_5dim, coef=0.05)  # Sigmoid function
        # activations = act_5dim
        #mins = neuralnet.np.min(act_5dim, 0)
        #maxs = neuralnet.np.max(act_5dim, 0)
        #print("maxs-mins:")
        #print((maxs - mins))
        #activations = (act_5dim - mins) / (maxs - mins)

        print(activations)
        client.send_message("/sound", activations[0])
