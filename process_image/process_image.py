"""

 process_image.py (author: Anson Wong / git: ankonzoid)

 Process your image using any of the following techniques:

  1) Force resizing (ypixels, xpixels) -> (ypixels_force, xpixels_force)

  2) Grey scaling (3 rgb channels -> 1 greyscale channel)

  3) K-means color quantization (using cluster colors or user-defined colors)

  4) Edge detection (gaussian blue, then sobel edge detection)

"""
import os
from src.img2kmeans import img2kmeans
from src.img2greyscale import img2greyscale
from src.img2resize import img2resize
from src.img2edges import img2edges

def process_image(settings, imgFile, outDir):

    # Create output directory
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    # Get file name without extension
    nametag = os.path.splitext(os.path.basename(imgFile))[0]

    # Process image
    print("Processing {} to {}...".format(imgFile, outDir))

    if "img2resize" in settings.keys():
        print("Resizing image...")
        img2resize(imgFile, os.path.join(outDir, nametag + "_resized.jpg"),
                   ypixels_resize=settings["img2resize"]["ypixels"],
                   xpixels_resize=settings["img2resize"]["xpixels"])

    if "img2greyscale" in settings.keys():
        print("Greyscaling image...")
        img2greyscale(imgFile, os.path.join(outDir, nametag + "_greyscale.jpg"))

    if "img2kmeans" in settings.keys():
        print("k-means quantizing image...")
        img2kmeans(imgFile, os.path.join(outDir, nametag + "_kmeans.jpg"),
                   k=settings["img2kmeans"]["k"],
                   use_custom_colors=settings["img2kmeans"]["use_custom_colors"],
                   custom_colors=settings["img2kmeans"]["custom_colors"])

    if "img2edges" in settings.keys():
        print("Sobel edge detecting image...")
        img2edges(imgFile, os.path.join(outDir, nametag + "_edges.jpg"))

settings = {
    "img2resize": {"ypixels": 100,
                   "xpixels": 100},
    "img2greyscale": {},
    "img2kmeans": {"k": 5,
                   "use_custom_colors": False,
                   "custom_colors": [[1, 1, 1],
                                     [37/255, 156/255, 247/255],
                                     [0, 0, 0]]},
    "img2edges": {},  # requires opencv
}

# Run our tools
process_image(settings=settings,
              imgFile=os.path.join(os.getcwd(), "input", "squirrel.jpeg"),
              outDir=os.path.join(os.getcwd(), "output"))
