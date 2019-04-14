import skimage.io
from skimage.transform import resize

def img2resize(input_filename, output_filename, ypixels=100, xpixels=100):
    img = skimage.io.imread(input_filename, as_gray=False)
    img_resized = resize(img, (ypixels, xpixels), anti_aliasing=True)
    skimage.io.imsave(output_filename, img_resized)