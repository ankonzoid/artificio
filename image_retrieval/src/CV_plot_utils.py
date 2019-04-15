"""

 CV_plot_utils.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn import manifold

# Plot image
def plot_img(img, range=[0, 255]):
    plt.imshow(img, vmin=range[0], vmax=range[1])
    plt.xlabel("xpixels")
    plt.ylabel("ypixels")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plots images in 2 rows: top row is query, bottom row is answer
def plot_query_retrieval(img_query, imgs_retrieval, outFile):
    n_retrieval = len(imgs_retrieval)
    fig = plt.figure(figsize=(2*n_retrieval, 4))
    fig.suptitle("Image Retrieval (k={})".format(n_retrieval), fontsize=25)

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
        ax.set_title("Rank #%d" % (i+1), fontsize=14)  # set subplot title

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
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=True)
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
            imscatter(X[:,0], X[:,1], imgs, zoom=0.3, ax=ax)

        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title, fontsize=18)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne, imgs, "t-SNE embeddings")
    if outFile is None:
        plt.show()
    else:
        plt.savefig(outFile, bbox_inches='tight')
    plt.close()

# Plot image reconstructions
def plot_reconstructions(imgs, imgs_reconstruct, outFile,
                         range_imgs=[0, 255],
                         range_imgs_reconstruct=[0, 1]):
    # Create plot to save
    assert len(imgs) == len(imgs_reconstruct)
    fig = plt.figure(figsize=(20, 4))
    fig.suptitle("Image Reconstructions", fontsize=35)
    n = min(len(imgs), 10)
    for i in range(n):

        # Plot original image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(imgs[i],
                   vmin=range_imgs[0],
                   vmax=range_imgs[1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Plot reconstructed image
        ax = plt.subplot(2, n, n + i + 1)
        plt.imshow(imgs_reconstruct[i],
                   vmin=range_imgs_reconstruct[0],
                   vmax=range_imgs_reconstruct[1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if outFile is None:
        plt.show()
    else:
        plt.savefig(outFile, bbox_inches='tight')
    plt.close()