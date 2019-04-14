import numpy as np
import skimage.io
from PIL import Image

def img2greyscale(input_filename, output_filename):
    img_pil = Image.open(input_filename)  # PIL object
    img_greyscale = np.asarray(img_pil.convert('L'))  # greyscale
    skimage.io.imsave(output_filename, img_greyscale)