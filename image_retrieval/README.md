# Image Retrieval (via Autoencoders / Transfer Learning)

Given a set of query images and database images, we perform image retrieval on database images to get the top-k most similar database images using kNN on the image embeddings with cosine similarity as the distance metric. As an example, we provide 36 steakhouse food images (6 of each food class: steak, potato, french fries, salads, burger, asparagus) and make similar image food recommendations for 3 unseen test images.

We provide two unsupervised methods of doing this: 

1) **Transfer learning** by performing generating image embeddings using a pre-trained network such as VGG19. This is done by removing its last few layers, and performing inference on our images vectors for the generation of flattened embeddings. No training is needed throughout this entire processing, only the loading of the pre-trained weights.

<p align="center"> 
<img src="coverart/TL_concept.jpg" width="60%">
</p>

2) **Training an autoencoder** (fully-connected or convolutional) on our database images to minimize the reconstruction loss. After sufficient training, we extract the encoder part of the autoencoder and use it during inference to generate flattened embeddings.

<p align="center"> 
<img src="coverart/AE_concept.jpg" width="60%">
</p>

<p align="center"> 
<img src="coverart/AE_reconstruction.png" width="60%">
</p>

We also provide visualizations of the image retrievals:

1) Transfer learning

<p align="center"> 
<img src="coverart/TL_rec_test_asparagus.png" width="50%">
</p>

<p align="center"> 
<img src="coverart/TL_rec_test_salad.png" width="50%">
</p>

<p align="center"> 
<img src="coverart/TL_rec_test_burger.png" width="50%">
</p>

1) Autoencoders

<p align="center"> 
<img src="coverart/AE_rec_test_asparagus.png" width="50%">
</p>

<p align="center"> 
<img src="coverart/AE_rec_test_salad.png" width="50%">
</p>

<p align="center"> 
<img src="coverart/AE_rec_test_burger.png" width="50%">
</p>

As well as t-SNE visualizations of the database image embeddings:

<p align="center"> 
<img src="coverart/TL_tsne.png" width="45%">
</p> 

### Usage

Run

```
python3 image_retrieval.py
```    

after adjusting parameters (`simpleAE` is simple FC autoencoder, `convAE` is multi-layer convolutional autoencoder, `vgg19` is pre-trained VGG19)

```
modelName = "convAE"  # try: "simpleAE", "convAE", "vgg19"
trainModel = True
saveModel = False
```

in `image_retrieval.py` to your purpose. All retrieval visualizations can be found in the `output` directory.

### Example output

```
Reading train images...
Reading test images...
Image shape = (100, 100, 3)
Loading VGG19 pre-trained model...
input_shape_model = (100, 100, 3)
output_shape_model = (3, 3, 512)
Applying image transformer to training images...
Applying image transformer to test images...
 -> X_train.shape = (36, 100, 100, 3)
 -> X_test.shape = (3, 100, 100, 3)
Inferencing embeddings using pre-trained model...
 -> E_train.shape = (36, 4608)
 -> E_test.shape = (3, 4608)
Fitting k-nearest-neighbour model on training images...
Performing image retrieval on test images...
Visualizing t-SNE on training images...
```

### Libraries

* tensorflow, skimage, sklearn, multiprocessing, numpy, matplotlib

### Authors

Anson Wong

