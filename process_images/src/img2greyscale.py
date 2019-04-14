import skimage.io

def img2greyscale(input_filename, output_filename):
    img_greyscale = skimage.io.imread(input_filename, as_gray=True)
    skimage.io.imsave(output_filename, img_greyscale)