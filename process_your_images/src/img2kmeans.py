"""

 img2kmeans.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

import scipy.misc
from PIL import Image

def img2kmeans(input_filename, output_filename,
               k=5, use_custom_colors=None, custom_colors=None, sort=True):

    # Read image
    img_pil = Image.open(input_filename)  # PIL object
    img = np.asarray(img_pil)  # numpy array

    img_norm = np.array(img, dtype=np.float64) / 255  # normalize r,g,b values
    w, h, d = tuple(img_norm.shape)
    assert d == 3

    # Choose k
    k_use = k
    if use_custom_colors:
        k_use = len(custom_colors)

    # Fit k-means model
    n_sample = 1000
    X = np.reshape(img_norm, (w * h, d))  # reshape image matrix to be 2D
    X_sample = shuffle(X, random_state=0)[:n_sample]  # take a shuffled sample of all pixels
    model = KMeans(n_clusters=k_use, random_state=0).fit(X_sample)
    y_pred = model.predict(X)

    # Create image from k-means result
    # use_custom_colors == False -> kmeans.cluster_centers_
    # use_custom_colors == True -> sorted custom_colors
    if not use_custom_colors:
        cluster_colors = model.cluster_centers_
    else:
        cluster_colors = choose_cluster_colors(model, custom_colors, sort=sort)

    # Create image based on these colors
    img_kmeans = recreate_image(cluster_colors, y_pred, w, h)

    # Save image
    scipy.misc.imsave(output_filename, img_kmeans)

def choose_cluster_colors(model, custom_colors, sort=True):

    # Return custom colors if no need to sort
    if sort == False:
        return custom_colors

    custom_colors = np.array(custom_colors, dtype=float)
    n_colors = len(custom_colors)

    # Sort k-means cluster centers by their euclidean length from (0,0,0)
    brightness_vec_user = np.zeros((n_colors), dtype=float)
    brightness_vec_kmeans = np.zeros((n_colors), dtype=float)

    for i in range(n_colors):
        rgb_vec_kmeans = model.cluster_centers_[i]
        rgb_vec_user = custom_colors[i]
        brightness_vec_user[i] = np.sum(rgb_vec_user)  # sum rgb pixels for brightness
        brightness_vec_kmeans[i] = np.sum(rgb_vec_kmeans)  # sum rgb pixels for brightness

    # Sort by ascending brightness
    i_brightness_user_sorted = np.argsort(brightness_vec_user)  # ascending brightness
    i_brightness_kmeans_sorted = np.argsort(brightness_vec_kmeans)  # ascending brightness

    # Build sorted custom colors
    customized_kmeans_cluster_centers = np.zeros((n_colors, 3), dtype=float)
    for i in range(n_colors):
        j = i_brightness_kmeans_sorted[i]
        k = i_brightness_user_sorted[i]
        customized_kmeans_cluster_centers[j] = custom_colors[k]

    cluster_colors = custom_colors

    return cluster_colors

def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image