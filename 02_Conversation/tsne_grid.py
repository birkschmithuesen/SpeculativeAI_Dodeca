import numpy as np
import os
import argparse
import glob

from PIL import Image
from lapjv import lapjv
from sklearn.manifold import TSNE
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from tensorflow.python.keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import GlobalAveragePooling2D


out_dim = 6  # number of images in a row/column in output image
in_dir = './camera_view/video_frames/'  # source directory for images
out_dir = './'  # destination directory for output image

out_name = 'out_image.jpg'  # name of output image file
to_plot = np.square(out_dim)
out_res = 224  # width/height of images in output square image
perplexity = 10  # TSNE perplexity - the TSNE is responsibly for projecting the multi dimensional vector into 2 dimensional space
tsne_iter = 5000000  # number of iterations in TSNE algorithm

if out_dim == 1:
    raise ValueError("Output grid dimension 1x1 not supported.")

if not os.path.exists(out_dir):
    raise argparse.ArgumentTypeError("'{}' not a valid directory.".format(out_dir))

if not os.path.exists(in_dir):
    raise argparse.ArgumentTypeError("'{}' not a valid directory.".format(in_dir))


def build_model():
    """
    This function load a trained VGG16 CNN excluding the final Dense layer. Then it adds one more layer 
    (GlobalAveragePooling2D) to transform the last volume (?,?,512) to a 512 dimensional vector
    :return: the model
    """
    base_model = VGG16(weights='imagenet', include_top=False)  # do not load final Dense Layers
    top_model = Sequential()
    top_model.add(GlobalAveragePooling2D())  # Transform Conv+Pooling in a vector (512 dimensions)
    return Model(inputs=base_model.input, outputs=top_model(base_model.output))


def load_img(input_dir):
    """ 
    This function load all PNG images stored in 'input_dir'
    :param input_dir: the directory where the images are stored
    :return: a list with the images and a list with the names of the files
    """
    pred_img = sorted(glob.glob(input_dir + "/*.png"))
    img_collection = []
    file_names = []
    for idx, img in enumerate(pred_img):
        print("Loading image", idx+1, ":", img)
        file_names.append(img)
        img_collection.append(image.load_img(img, target_size=(out_res, out_res)))
    if np.square(out_dim) > len(img_collection):
        raise ValueError("Cannot fit {} images in {}x{} grid".format(len(img_collection), out_dim, out_dim))
    return img_collection, file_names


def get_activations(model, img_collection, file_names):
    """ 
    This function computes one vector for each image using the CNN model and store them in 'activations'.
    Each column of 'activations' corresponds with the vector of one image. 
    Those vectors are stored in 'out_vectors.csv'.
    This function also produces the file 'out_distances.csv' where the pairwise distances between those vectos are
    write down.
    :param model: the trained CNN model
    :param img_collection: the list with the loaded images
    :param file_names: the list with the corresponding file names of the images
    :return: a list with the vectors and the files described above
    """
    activations = []
    header = ''
    for idx, img in enumerate(img_collection):
        # if idx == to_plot:
        #     break
        print("Processing image", idx+1, ":", img)
        img = img.resize((out_res, out_res), Image.ANTIALIAS)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        activations.append(np.squeeze(model.predict(x)))
        header = header + os.path.basename(file_names[idx]).split('.')[0] + ';'
    np.savetxt("out_vectors.csv", list(map(list, zip(*activations))), delimiter=";", header=header)
    distances = distance_matrix(activations, activations)
    np.savetxt("out_distances.csv", distances, delimiter=";", header=header)
    return activations


def generate_tsne(activations):
    """
    This function maps the vectors in 'activations' in a 2 dimensional space where those vectors that similar must 
    appear close. It uses the algorithm TSNE.
    :param activations: a list with vectors 
    :return: the representation of those vector in that 2 dimensional space
    """
    tsne = TSNE(perplexity=perplexity, n_components=2, init='random', n_iter=tsne_iter)
    x_2d = tsne.fit_transform(np.array(activations)[0:to_plot, :])
    x_2d -= x_2d.min(axis=0)
    x_2d /= x_2d.max(axis=0)
    return x_2d


def save_tsne_grid(img_collection, x_2d, output_res, output_dim, output_dir):
    """
    This function creates an image made of the original images placed relative to what is indicated in x_2d
    :param img_collection: the list with the loaded images
    :param x_2d: the 2d coordinates obtained for the images using TSNE
    :param output_res: resolution of the images
    :param output_dim: the composed image will be created using (output_dim x output_dim) images
    :param output_dir: destination directory for output image
    :return: This function produces a JPG image in the 'output_dir' directory
    """
    grid = np.dstack(np.meshgrid(np.linspace(0, 1, output_dim), np.linspace(0, 1, output_dim))).reshape(-1, 2)
    cost_matrix = cdist(grid, x_2d, "sqeuclidean").astype(np.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    row_asses, col_asses, _ = lapjv(cost_matrix)
    grid_jv = grid[col_asses]
    out = np.ones((output_dim * output_res, output_dim * output_res, 3))

    for pos, img in zip(grid_jv, img_collection[0:to_plot]):
        h_range = int(np.floor(pos[0] * (output_dim - 1) * output_res))
        w_range = int(np.floor(pos[1] * (output_dim - 1) * output_res))
        out[h_range:h_range + output_res, w_range:w_range + output_res] = image.img_to_array(img)

    im = image.array_to_img(out)
    im.save(output_dir + out_name, quality=100)


def main():
    model = build_model()
    model.summary()

    img_collection, names_of_file = load_img(in_dir)

    activations = get_activations(model, img_collection, names_of_file)

    # Birk, from here it is only for generating the image. Actually, you do not need it but it is very cool to see it
    # and it is useful in order to understand the nature of the vectors that the CNN produce
    print("Generating 2D representation.")
    x_2dim = generate_tsne(activations)
    print("Generating image grid.")
    save_tsne_grid(img_collection, x_2dim, out_res, out_dim, out_dir)

if __name__ == '__main__':
    main()
