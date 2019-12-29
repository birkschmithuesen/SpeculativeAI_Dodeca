import numpy as np
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from conversation import neuralnet_vision

TENSORRT_MODEL_PATH = "data/TensorRT_model.pb"

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

