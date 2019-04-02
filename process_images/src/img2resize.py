"""
 convert_img2resize.py
"""
import numpy as np
import scipy.misc
from PIL import Image

def img2resize(input_filename, output_filename, ypixels=100, xpixels=100):
    img_pil = Image.open(input_filename)  # PIL object
    img = np.asarray(img_pil)  # numpy array
    img_resized = scipy.misc.imresize(img, (ypixels, xpixels))  # resize
    scipy.misc.imsave(output_filename, img_resized)  # save