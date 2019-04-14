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
from src.models.autoencoder import AutoEncoder
from src.utils import makeDir
from src.CV_IO_utils import read_imgs_dir
from src.CV_transform_utils import apply_transformer
from src.CV_transform_utils import resize_img, normalize_img, flatten_img
from src.CV_plot_utils import plot_query_retrieval, plot_tsne

# Run mode
modelName = "vgg19" # simpleAE, convAE, vgg19
trainModel = True
saveModels = False

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
shape_img = imgs_train[0].shape
print("Image shape = {}".format(shape_img))

# Build models
if modelName in ["simpleAE", "convAE"]:

    info = {
        "shape_img": shape_img,
        "autoencoderFile": os.path.join(outPath, "{}_autoecoder.h5".format(modelName)),
        "encoderFile": os.path.join(outPath, "{}_encoder.h5".format(modelName)),
        "decoderFile": os.path.join(outPath, "{}_decoder.h5".format(modelName)),
    }

    # Set up autoencoder base class
    model = AutoEncoder(modelName, info)
    model.set_arch()

    if modelName == "simpleAE":
        shape_img_resize = shape_img
        input_shape_model = (model.encoder.input.shape[1],)
        output_shape_model = (model.encoder.output.shape[1],)
        n_epochs = 100
    elif modelName == "convAE":
        shape_img_resize = shape_img
        input_shape_model = tuple([int(x) for x in model.encoder.input.shape[1:]])
        output_shape_model = tuple([int(x) for x in model.encoder.output.shape[1:]])
        n_epochs = 200
    else:
        raise Exception("Invalid modelName!")

elif modelName in ["vgg19"]:

    # Load pre-trained VGG19 model + higher level layers
    print("Loading VGG19 pre-trained model...")
    model = tf.keras.applications.VGG19(weights='imagenet', include_top=False,
                                        input_shape=shape_img)
    model.summary()

    shape_img_resize = tuple([int(x) for x in model.input.shape[1:]])
    input_shape_model = tuple([int(x) for x in model.input.shape[1:]])
    output_shape_model = tuple([int(x) for x in model.output.shape[1:]])
    n_epochs = None

else:
    raise Exception("Invalid modelName!")

# Print some model info
print("input_shape_model = {}".format(input_shape_model))
print("output_shape_model = {}".format(output_shape_model))

# Apply transformations to all images
class ImageTransformer(object):

    def __init__(self, shape_resize):
        self.shape_resize = shape_resize

    def __call__(self, img):
        img_transformed = resize_img(img, self.shape_resize)
        img_transformed = normalize_img(img_transformed)
        return img_transformed

transformer = ImageTransformer(shape_img_resize)
print("Applying image transformer to training images...")
imgs_train_transformed = apply_transformer(imgs_train, transformer, parallel=True)
print("Applying image transformer to test images...")
imgs_test_transformed = apply_transformer(imgs_test, transformer, parallel=True)

# Convert images to numpy array
X_train = np.array(imgs_train_transformed).reshape((-1,) + input_shape_model)
X_test = np.array(imgs_test_transformed).reshape((-1,) + input_shape_model)
print(" -> X_train.shape = {}".format(X_train.shape))
print(" -> X_test.shape = {}".format(X_test.shape))

# Train (if necessary)
if modelName in ["simpleAE", "convAE"]:
    if trainModel:
        model.compile(loss="binary_crossentropy", optimizer="adam")
        model.fit(X_train, n_epochs=300, batch_size=256)
        if saveModels:
            model.save_models()
    else:
        model.load_models()

# Create embeddings using model
print("Inferencing embeddings using pre-trained model...")
E_train = model.predict(X_train).reshape((-1, np.prod(list(output_shape_model))))
E_test = model.predict(X_test).reshape((-1, np.prod(list(output_shape_model))))
print(" -> E_train.shape = {}".format(E_train.shape))
print(" -> E_test.shape = {}".format(E_test.shape))

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