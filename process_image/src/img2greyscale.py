import skimage.io

def img2greyscale(imgFile, outFile):
    img_greyscale = skimage.io.imread(imgFile, as_gray=True)
    skimage.io.imsave(outFile, img_greyscale)