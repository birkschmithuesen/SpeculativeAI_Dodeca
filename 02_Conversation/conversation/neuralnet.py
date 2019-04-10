import numpy as np
import os
import argparse
import glob
from joblib import dump, load

from PIL import Image
from lapjv import lapjv
from sklearn.manifold import TSNE
from scipy.spatial import distance_matrix
# from scipy.misc import imsave
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from tensorflow.python.keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import GlobalAveragePooling2D

GENERATE_BIG_PICTURE = True
TRANSFORM_FROM_CNN_DIM_TO_SOUND_DIM = True
TRANSFORM_USING_PCA = True  # True: transform form 512 to 5 using PCA. Otherwise, using random matrix
WORK_WITH_ORIGINAL_IMAGES = True  # True: compute the 512d vector from de original image. Otherwise, from Black/White
CNN_DIM = 512
SOUND_DIM = 5

out_dim = 6  # number of images in a row/column in output image
in_dir = '../camera_view/video_frames/'  # source directory for images
out_dir = '../'  # destination directory for output image


to_plot = np.square(out_dim)
out_res = 224  # width/height of images in output square image
perplexity = 10  # TSNE perplexity
tsne_iter = 5000000  # number of iterations in TSNE algorithm

if out_dim == 1:
    raise ValueError("Output grid dimension 1x1 not supported.")

if __name__ == '__main__':    
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
        print("Loading image", idx + 1, ":", img)
        file_names.append(img)
        img_collection.append(image.load_img(img, target_size=(out_res, out_res)))
    if np.square(out_dim) > len(img_collection):
        raise ValueError("Cannot fit {} images in {}x{} grid".format(len(img_collection), out_dim, out_dim))
    return img_collection, file_names


def binarize_array(numpy_array, threshold=200):
    """Binarize an image in a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array


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
    :return: a list with the vectors, the files described above and the PIL images in B/W
    """
    activations = []
    header = ''
    img_collection_bn = []
    for idx, img in enumerate(img_collection):
        # if idx == to_plot:
        #     break
        print("Processing image", idx + 1, ":", img)
        img = img.resize((out_res, out_res), Image.ANTIALIAS)
        img_bw = img.convert('L')  # convert image to monochrome
        xbw = image.img_to_array(img_bw)
        xbw = np.reshape(xbw, (224, 224))
        xbw = binarize_array(xbw)  # filtering
        img_bw = Image.fromarray(xbw)
        img_bw = img_bw.convert('RGB')  # convert image to RGB
        img_bw.save(file_names[idx]+'.BW.jpg')
        img_collection_bn.append(img_bw)
        if WORK_WITH_ORIGINAL_IMAGES:
            x = image.img_to_array(img)
        else:
            x = image.img_to_array(img_bw)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        activations.append(np.squeeze(model.predict(x)))
        header = header + os.path.basename(file_names[idx]).split('.')[0] + ';'
    np.savetxt("out_vectors.csv", list(map(list, zip(*activations))), delimiter=";", header=header)
    distances = distance_matrix(activations, activations)
    np.savetxt("out_distances.csv", distances, delimiter=";", header=header)
    return activations, header, img_collection_bn


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


def save_tsne_grid(img_collection, x_2d, output_res, output_dim, output_dir, out_name):
    """
    This function creates an image made of the original images placed relative to what is indicated in x_2d
    :param img_collection: the list with the loaded images
    :param x_2d: the 2d coordinates obtained for the images using TSNE
    :param output_res: resolution of the images
    :param output_dim: the composed image will be created using (output_dim x output_dim) images
    :param output_dir: destination directory for output image
    :param out_name: name of the output image
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


def generate_M_and_MG(seed, deviation):
    """
    This function generates random matrices M and MG
    :param seed: seed for the random number generator
    :param deviation: for generate following a normal distribution. Small deviation in order to generate numbers
    close to cero
    :return: matrices M and MG
    """
    np.random.seed(seed=seed)
    M = np.random.uniform(low=-1, high=1, size=(CNN_DIM, SOUND_DIM))
    MG = np.random.normal(0, scale=deviation, size=(CNN_DIM, SOUND_DIM))
    return M, MG


def get_M_and_MG_from_file():
    """
    This function loads both matrices from files
    :return: matrices M and MG
    """
    M = np.load('matrixUniform.npy')
    MG = np.load('matrixGaussian.npy')
    return M, MG


def get_M_from_file_and_generate_MG(seed, deviation):
    """
    This function loads M from file and generates MG
    :param seed: seed for the random number generator
    :param deviation: for generate following a normal distribution. Small deviation in order to generate numbers
    close to cero
    :return: matrices M and MG
    """
    np.random.seed(seed=seed)
    M = np.load('matrixUniform.npy')
    MG = np.random.normal(0, scale=deviation, size=(CNN_DIM, SOUND_DIM))
    return M, MG


def sigmoid(mat, coef):
    """
    This function uses sigmoid function to map all values in 'mat' in the range [0,1]
    :param mat: a list of vectors. In this case each vector represents one image
    :param coef: modify the slope of the sigmoid: big values of coef make the function shrink
    :return: the matrix with all the values in [0,1]
    """
    return 1 / (1 + np.exp(-mat * coef))  # Sigmoid function


def main():
    model = build_model()
    model.summary()

    img_collection, names_of_file = load_img(in_dir)

    activations, header, img_coll_bn = get_activations(model, img_collection, names_of_file)

    if TRANSFORM_FROM_CNN_DIM_TO_SOUND_DIM:
        if TRANSFORM_USING_PCA:
            # you can comment the following 3 lines when you have trained the PCA previously
            pca = PCA(n_components=SOUND_DIM)
            act_5dim = pca.fit_transform(activations)
            dump(pca, 'pca.joblib')  # store the trained pca

            # uncomment the following 2 lines if you have trained previously de PCA
            # pca = load('pca.joblib')
            # act_5dim = pca.transform(activations)

            # if you want to add some gaussian noise to the 5dim vectors you can uncomment the following 2 lines
            # MG = np.random.normal(0, scale=0.1, size=act_5dim.shape)
            # act_5dim = act_5dim + MG
        else:
            # M: is a random (uniform dist.) matrix that projects CNN_DIM dimensional vectors in SOUND_DIM dim
            # MG: is a random (normal dist.) matrix if you want to slightly modify M
            # comment/uncomment next lines to obtain what you want
            # change the seed to obtain different matrices
            M, MG = generate_M_and_MG(seed=2022, deviation=0.01)
            # M, MG = get_M_and_MG_from_file()
            # M, MG = get_M_from_file_and_generate_MG(seed=2019, deviation=0.01)

            # you can modify M a little bit uncommenting next line
            # M = M + MG

            # if you want to store the matrices in files you need to uncomment next lines
            # Be careful!!! if the matrices already exist then these instructions will override them
            np.save('matrixUniform', M)
            np.save('matrixGaussian', MG)

            act_5dim = activations @ M  # matrix multiplication

        # activations = sigmoid(act_5dim, coef=0.05)  # Sigmoid function
        # activations = act_5dim
        mins = np.min(act_5dim, 0)
        maxs = np.max(act_5dim, 0)
        activations = (act_5dim - mins) / (maxs - mins)

        # store in files the new 5dim vectors and the pairwise distances
        np.savetxt("out_vectors_5d.csv", list(map(list, zip(*activations))), delimiter=";", header=header)
        distances = distance_matrix(activations, activations)
        np.savetxt("out_distances_5d.csv", distances, delimiter=";", header=header)

    if GENERATE_BIG_PICTURE:
        # Birk, from here it is only for generating the image. Actually, you do not need it but it is very cool to see
        # it and it is useful in order to understand the nature of the vectors that the CNN produce
        print("Generating 2D representation.")
        x_2dim = generate_tsne(activations)
        print("Generating image grid.")
        save_tsne_grid(img_collection, x_2dim, out_res, out_dim, out_dir, out_name='out_image_original.jpg')
        save_tsne_grid(img_coll_bn, x_2dim, out_res, out_dim, out_dir, out_name='out_image_BW.jpg')


if __name__ == '__main__':
    main()
