import numpy as np
import skimage.io
import cv2

def img2edges(imgFile, outFile):

    img = skimage.io.imread(imgFile, as_gray=False)

    # Blur the image with a Gaussian kernel
    kernel_size = (5, 5)
    img_blurred = img.copy() if kernel_size == (0,0) else cv2.GaussianBlur(img.copy(), kernel_size, 0)

    # Find the edges from the blurred image
    edges = np.max(np.array([edgedetect(img_blurred[:, :, 0]),
                             edgedetect(img_blurred[:, :, 1]),
                             edgedetect(img_blurred[:, :, 2])]), axis=0)  # Sobel edge detection

    # Process edges by zero-ing out pixels less than the mean
    edges_zeroed = edges.copy()
    edges_zeroed[edges.copy() <= np.mean(edges.copy())] = 0  # zero out pixels less than mean
    img_edges = edges_zeroed

    # Save
    skimage.io.imsave(outFile, img_edges)

# edgedetect:
def edgedetect(channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255  # Some values seem to go above 255. However RGB channels has to be within 0-255
    return sobel
