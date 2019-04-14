"""

 image_retrieval.py  (author: Anson Wong / git: ankonzoid)

 We perform image retrieval using transfer learning on a pre-trained
 VGG image classifier. We plot the k=5 most similar images to our
 query images, as well as the t-SNE visualizations.

"""
import os
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from image_retrieval.src.utils import makeDir
from image_retrieval.src.CV_io_utils import read_imgs_dir
from image_retrieval.src.CV_transform_utils import apply_transformer
from image_retrieval.src.CV_transform_utils import normalize_img, resize_img
from image_retrieval.src.CV_plot_utils import plot_query_retrieval, plot_tsne

# Run mode
modelName = "simpleAE"
if modelName == "simpleAE": # autoencoder
    trainModel = True

elif modelName == "convAE": # autoencoder
    trainModel = True

elif modelName == "vgg19": # transfer learning
    trainModel = False

else:
    raise Exception("modelName of '{}' given!".format(modelName))

# Make paths
dataTrainPath = os.path.join(os.getcwd(), "data", "train")
dataTestPath = os.path.join(os.getcwd(), "data", "test")
outPath = makeDir(os.path.join(os.getcwd(), "output", modelName))

# Read images
extensions = [".jpg", ".jpeg"]
print("Reading train images from '{}'...".format(dataTrainPath))
imgs_train = read_imgs_dir(dataTrainPath, extensions, parallel=True)
print("Reading test images from '{}'...".format(dataTestPath))
imgs_test = read_imgs_dir(dataTestPath, extensions, parallel=True)

# Load pre-trained VGG19 model + higher level layers
print("Loading VGG19 pre-trained model...")
model = tf.keras.applications.VGG19(weights='imagenet', include_top=False,
                                    input_shape=(224, 224, 3))
input_shape_model = tuple([int(x) for x in model.input.shape[1:]])
output_shape_model = tuple([int(x) for x in model.output.shape[1:]])

# Apply transformations
class ImageTransformer(object):

    def __init__(self, shape_resize):
        self.shape_resize = shape_resize

    def __call__(self, img):
        img_transformed = resize_img(img, self.shape_resize)
        img_transformed = normalize_img(img_transformed)
        return img_transformed

transformer = ImageTransformer(input_shape_model)
print("Applying image transformer to training images...")
imgs_train_transformed = apply_transformer(imgs_train, transformer, parallel=True)
print("Applying image transformer to test images...")
imgs_test_transformed = apply_transformer(imgs_test, transformer, parallel=True)

# Convert images to data for neural nets
X_train = np.array(imgs_train_transformed).reshape((-1,) + input_shape_model)
X_test = np.array(imgs_test_transformed).reshape((-1,) + input_shape_model)
print(" -> X_train.shape = {}".format(X_train.shape))
print(" -> X_test.shape = {}".format(X_test.shape))

# Train embedding model
if trainModel:
    model.fit(X_train)

# Create embeddings using pre-trained model
print("Inferencing embeddings using pre-trained model...")
E_train = model.predict(X_train).reshape((-1, np.prod(list(output_shape_model))))
E_test = model.predict(X_test).reshape((-1, np.prod(list(output_shape_model))))
print(" -> EMB_train.shape = {}".format(E_train.shape))
print(" -> EMB_test.shape = {}".format(E_test.shape))

# Fit kNN model on training images
print("Fitting k-nearest-neighbour model on training images...")
knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(E_train)

# Perform image retrieval on test images
print("Performing image retrieval on test images...")
for i, emb in enumerate(E_test):
    _, indices = knn.kneighbors([emb]) # find k nearest train neighbours
    img_query = imgs_test[i] # query image
    imgs_retrieval = [imgs_train[idx] for idx in indices.flatten()] # retrieval images
    outFile = os.path.join(outPath, "retrieval_{}.png".format(i))
    plot_query_retrieval(img_query, imgs_retrieval, outFile) # plot

# Plot t-SNE visualization
print("Visualizing t-SNE on training images...")
outFile = os.path.join(outPath, "tsne.png")
plot_tsne(E_train, imgs_train, outFile)