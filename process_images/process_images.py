"""

 process_images.py (author: Anson Wong / git: ankonzoid)

 Process your images in the `input` directory using any of the following techniques.
 The results

 Standard techniques:

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

def process_your_images(settings, inPath, outPath):

    # Check input and output directories exist
    if not os.path.isdir(inPath):
        exit("err: could not find input directory '{}'".format(inPath))
    if not os.path.isdir(outPath):
        exit("err: could not find output directory '{}'".format(outPath))

    # Process each image in the input directory
    files_input_dir = os.listdir(inPath)
    n_files = len(files_input_dir)
    for i, file in enumerate(files_input_dir):  # extract local filenames

        # Consider only files that end with .jpg and .jpeg
        if not file.endswith((".jpg", ".jpeg")):
            continue

        # Build filename and nametag for output
        img_filename = os.path.join(inPath, file)
        nametag = os.path.splitext(file)[0]

        # Process image
        print("[{}/{}] Processing '{}'...".format(i+1, n_files, file))

        if settings["img2resize"]["use"]:
            print("Resizing image...")
            output_filename = os.path.join(outPath, nametag + "_resized.jpg")
            img2resize(img_filename, output_filename,
                       ypixels=settings["img2resize"]["ypixels"],
                       xpixels=settings["img2resize"]["xpixels"])

        if settings["img2greyscale"]["use"]:
            print("Greyscaling image...")
            output_filename = os.path.join(outPath, nametag + "_greyscale.jpg")
            img2greyscale(img_filename, output_filename)

        if settings["img2kmeans"]["use"]:
            print("k-means quantizing image...")
            output_filename = os.path.join(outPath, nametag + "_kmeans.jpg")
            img2kmeans(img_filename, output_filename,
                       k=settings["img2kmeans"]["k"],
                       use_custom_colors=settings["img2kmeans"]["use_custom_colors"],
                       custom_colors=settings["img2kmeans"]["custom_colors"])

        if settings["img2edges"]["use"]:
            print("Sobel edge detecting image...")
            output_filename = os.path.join(outPath, nametag + "_edges.jpg")
            img2edges(img_filename, output_filename)

settings = {
    "img2resize": {"use": True,
                   "ypixels": 100,
                   "xpixels": 100},
    "img2greyscale": {"use": True},
    "img2kmeans": {"use": True,
                   "k": 5,
                   "use_custom_colors": False,
                   "custom_colors": [[1, 1, 1],
                                     [37/255, 156/255, 247/255],
                                     [0, 0, 0]]},
    "img2edges": {"use": True},  # requires opencv
}

# Run our tools
process_your_images(settings,
                    inPath=os.path.join(os.getcwd(), "input"),
                    outPath=os.path.join(os.getcwd(), "output"))
