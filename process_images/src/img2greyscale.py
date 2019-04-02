"""

 img2greyscale.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import scipy.misc
from PIL import Image

def img2greyscale(input_filename, output_filename):
    img_pil = Image.open(input_filename)  # PIL object
    img_greyscale = np.asarray(img_pil.convert('L'))  # greyscale
    scipy.misc.imsave(output_filename, img_greyscale)  # save