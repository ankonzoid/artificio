import skimage.io
from skimage.transform import resize

def img2resize(imgFile, outFile, ypixels_resize, xpixels_resize):
    img = skimage.io.imread(imgFile, as_gray=False)
    img_resized = resize(img, (ypixels_resize, xpixels_resize), anti_aliasing=True)
    skimage.io.imsave(outFile, img_resized)