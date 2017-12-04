"""

 sim_img_TL.py  (author: Anson Wong / git: ankonzoid)

 Uses transfer learning on pre-trained VGG image classification models to
 get feature vectors and plot the tSNE of the feature vectors.

"""
import numpy as np
import sys, os
sys.path.append("src")
from vgg16 import VGG16
from vgg19 import VGG19
from keras.preprocessing import image
from keras.models import Model
from imagenet_utils import preprocess_input
from tSNE import run_tsne
from KNN import KNearestNeighbours
from plot_utilities import PlotUtils
from sort_utilities import find_topk_unique

def main():
    # ================================================
    # Set pre-trained model
    # ================================================
    print()
    if 0:
        # Remove last layer, to get multiple filters
        print("Using VGG16 pre-trained model...")
        model = VGG16(weights='imagenet',
                      include_top=False)  # remove output layer
    else:
        print("Using VGG19 pre-trained model...")
        base_model = VGG19(weights='imagenet')
        model = Model(input=base_model.input,
                      output=base_model.get_layer('block4_pool').output)

    # ================================================
    # Read images and convert them to feature vectors
    # ================================================
    imgs_plot, heads_plot, X = [], [], []
    path = "db"
    print("Reading images from '{}'...".format(path))
    for f in os.listdir(path):
        # Process filename
        filename = os.path.splitext(f)  # filename in directory
        filename_full = os.path.join(path,f)  # full path filename
        head, ext = filename[0], filename[1]
        if ext.lower() not in [".jpg", ".jpeg"]:
            continue

        # Read image file
        img = image.load_img(filename_full, target_size=(224, 224))  # load
        imgs_plot.append(np.array(img))  # image
        heads_plot.append(head)  # filename head

        # Pre-process for model input
        img = image.img_to_array(img)  # convert to array
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = model.predict(img).flatten()  # features
        X.append(features)  # append feature extractor

    # Find forced resized pixels
    n_imgs = len(imgs_plot)
    ypixels = imgs_plot[0].shape[0]
    xpixels = imgs_plot[0].shape[1]

    # Convert
    X = np.array(X)  # feature vectors
    imgs_plot = np.array(imgs_plot)  # images
    print(" X_features = {}".format(X.shape))
    print(" imgs_plot = {}".format(imgs_plot.shape))

    # ===========================
    # Find k-nearest images to each image
    # ===========================
    # Train kNN
    n_neighbours = 5
    knn = KNearestNeighbours()
    knn.compile(n_neighbors=n_neighbours,
                algorithm="brute",
                metric="cosine")
    knn.fit(X)

    # Plot
    PU = PlotUtils()
    for ind_query in range(n_imgs):
        # Find top-k closest images in the feature space database to each image
        print("[{}/{}] Plotting similar image recommendations for: {}".format(ind_query+1, n_imgs, heads_plot[ind_query]))
        distances, indices = knn.predict(np.array([X[ind_query]]))
        distances = distances.flatten()
        indices = indices.flatten()
        indices, distances = find_topk_unique(indices, distances, n_neighbours)

        # Plot recommendations
        result_filename = os.path.join("output", "sim_imgs", "{}_rec.png".format(heads_plot[ind_query]))
        x_query_plot = imgs_plot[ind_query].reshape((-1, ypixels, xpixels, 3))
        x_answer_plot = imgs_plot[indices].reshape((-1, ypixels, xpixels, 3))
        PU.plot_query_answer(x_query=x_query_plot,
                             x_answer=x_answer_plot,
                             filename=result_filename)

    # ===========================
    # Plot tSNE
    # ===========================
    print("Plotting tSNE to output/tsne.png...")
    run_tsne(imgs_plot, X, "output/tsne.png")

# Driver
if __name__ == "__main__":
    main()