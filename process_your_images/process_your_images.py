"""

 process_your_images.py (author: Anson Wong / git: ankonzoid)

 Process your images in the `input` directory using any of the following techniques.
 The results

 Standard techniques:

  1) Force resizing (ypixels, xpixels) -> (ypixels_force, xpixels_force)

  2) Grey scaling (3 rgb channels -> 1 greyscale channel)

  3) K-means color quantization (using cluster colors or user-defined colors)

  4) Edge detection (gaussian blue, then sobel edge detection)

 Extra techniques (requires opencv):

  1) object crop (opencv for contour search, then force resize to object binding box)

  2) object binarization

"""
import os

from src.img2resize import img2resize
from src.img2greyscale import img2greyscale
from src.img2kmeans import img2kmeans
from src.img2edges import img2edges

def main():

    settings = {
        "img2resize": {"use": True, "ypixels": 100, "xpixels": 100},
        "img2greyscale": {"use": True},
        "img2kmeans": {"use": False, "k": 5, "use_custom_colors": False,
                       "custom_colors": [[1, 1, 1],
                                         [37/255, 156/255, 247/255],
                                         [0, 0, 0]]},
        "img2edges": {"use": True},  # requires opencv
    }

    # Run our tools
    process_your_images(settings, input_dir="input", output_dir="output")


def process_your_images(settings, input_dir="input", output_dir="output"):

    # Check input and output directories exist
    if not os.path.isdir(input_dir):
        exit("err: could not find input directory '{}'".format(input_dir))
    if not os.path.isdir(output_dir):
        exit("err: could not find output directory '{}'".format(output_dir))

    # Process each image in the input directory
    files_input_dir = os.listdir(input_dir)
    n_files = len(files_input_dir)
    for i, file in enumerate(files_input_dir):  # extract local filenames

        # Consider only files that end with .jpg and .jpeg
        if not file.endswith((".jpg", ".jpeg")):
            continue

        # Build filename and nametag for output
        img_filename = os.path.join(input_dir, file)
        nametag = os.path.splitext(file)[0]

        # Process image
        print("[{}/{}] Processing '{}'...".format(i+1, n_files, file))

        if settings["img2resize"]["use"]:
            output_filename = os.path.join(output_dir, nametag + "_resized.jpg")
            img2resize(img_filename, output_filename,
                       ypixels=settings["img2resize"]["ypixels"],
                       xpixels=settings["img2resize"]["xpixels"])

        if settings["img2greyscale"]["use"]:
            output_filename = os.path.join(output_dir, nametag + "_greyscale.jpg")
            img2greyscale(img_filename, output_filename)

        if settings["img2kmeans"]["use"]:
            output_filename = os.path.join(output_dir, nametag + "_kmeans.jpg")
            img2kmeans(img_filename, output_filename,
                       k=settings["img2kmeans"]["k"],
                       use_custom_colors=settings["img2kmeans"]["use_custom_colors"],
                       custom_colors=settings["img2kmeans"]["custom_colors"])

        if settings["img2edges"]["use"]:
            output_filename = os.path.join(output_dir, nametag + "_edges.jpg")
            img2edges(img_filename, output_filename)

# Driver
if __name__ == '__main__':
    main()
