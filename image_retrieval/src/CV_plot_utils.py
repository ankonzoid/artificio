"""

 CV_plot_utils.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn import manifold

# Plot image
def plot_img(img):
    plt.imshow(img)
    plt.xlabel("xpixels")
    plt.ylabel("ypixels")
    plt.tight_layout()
    plt.show()

# Plots images in 2 rows: top row is query, bottom row is answer
def plot_query_retrieval(img_query, imgs_retrieval, outFile):
    n_retrieval = len(imgs_retrieval)
    plt.figure(figsize=(2*n_retrieval, 4))

    # Plot query image
    ax = plt.subplot(2, n_retrieval, 0 + 1)
    plt.imshow(img_query)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(4)  # increase border thickness
        ax.spines[axis].set_color('black')  # set to black
    ax.set_title("query",  fontsize=14)  # set subplot title

    # Plot retrieval images
    for i, img in enumerate(imgs_retrieval):
        ax = plt.subplot(2, n_retrieval, n_retrieval + i + 1)
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1)  # set border thickness
            ax.spines[axis].set_color('black')  # set to black
        ax.set_title("retrieved #%d" % (i+1), fontsize=14)  # set subplot title

    if outFile is None:
        plt.show()
    else:
        plt.savefig(outFile, bbox_inches='tight')
    plt.close()

# Plot t-SNE of images
def plot_tsne(X, imgs, outFile):

    def imscatter(x, y, images, ax=None, zoom=1.0):
        if ax is None:
            ax = plt.gca()
        x, y = np.atleast_1d(x, y)
        artists = []
        for x0, y0, img0 in zip(x, y, images):
            im = OffsetImage(img0, zoom=zoom)
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
        return artists

    def plot_embedding(X, imgs, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure()
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], ".", fontdict={'weight': 'bold', 'size': 9})
        if hasattr(offsetbox, 'AnnotationBbox'):
            imscatter(X[:,0], X[:,1], imgs, zoom=0.1, ax=ax)

        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne, imgs, "t-SNE embedding of images")
    if outFile is None:
        plt.show()
    else:
        plt.savefig(outFile, bbox_inches='tight')
    plt.close()

# Plot
def plot_save_reconstruction(x_data_test, img_shape, n_plot=10):

    ypixels = img_shape[0]
    xpixels = img_shape[1]
    n_channels = 3

    # Extract the subset of test data to reconstruct and plot
    n = min(len(x_data_test), n_plot)
    x_data_test_plot = x_data_test[np.arange(n)]

    # Perform reconstructions on test data
    x_data_test_plot_reconstructed = self.encode_decode(x_data_test_plot)

    # Create plot to save
    plt.figure(figsize=(20, 4))
    for i in range(n):

        # Plot original image
        ax = plt.subplot(2, n, i + 1)
        img_show_test = \
            x_data_test_plot[i].reshape((ypixels, xpixels, n_channels))
        plt.imshow(img_show_test)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Plot reconstructed image
        ax = plt.subplot(2, n, n + i + 1)
        img_show_test_reconstructed = \
            x_data_test_plot_reconstructed[i].reshape((ypixels, xpixels, n_channels))
        plt.imshow(img_show_test_reconstructed)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if 0:
        plt.show()