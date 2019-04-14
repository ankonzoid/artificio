# Image Processing Tools

<p align="center">
<img src="coverart/coverart.jpg" width="90%">
</p>

Here we provide tools for you for processing your image dataset. All you need to do is place your .jpg or .jpeg images into the `input` directory, run the code, then pick up your processed images in the `output` directory.

Currently we provide 4 processing techniques for you to apply to your images:

1) Greyscaling

2) Force resizing

3) K-means color quantization based (cluster center colors or custom colors)

4) Edge detection (Gaussian blur + Sobel)



### Usage

Run

```
python3 process_images.py
```

### Example output

#### 1) Greyscaling

<img src="coverart/example_greyscale.jpg" width="45%" align="center">

#### 2) Force resizing (image is downsampled here)

<img src="coverart/example_resize.jpg" width="45%" align="center">

#### 3) K-means quantization (cluster centers colors or custom colors)

<img src="coverart/example_kmeans.jpg" width="45%" align="center">

#### 4) Edge detection (Gaussian blur + Sobel)

<img src="coverart/example_edges.jpg" width="45%" align="center">


### Libraries

* numpy
* Pillow
* matplotlib
* opencv
* skimage

### Authors

Anson Wong