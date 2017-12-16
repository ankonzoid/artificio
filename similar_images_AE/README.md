# Similar images finder (using Autoencoders)

Given a set of query images and a set of store inventory images, we find the top-k similar inventory images that are the most 'similar' to the set of query images in an unsupervised way of training an autoencoder, then using its encoder to embed the images and perform kNN in to find 'similar' images. 

<p align="center"> 
<img src="https://github.com/ankonzoid/artificio/blob/master/similar_images_AE/coverart/coverart.jpg" width="60%">
</p>

In this code, we train a convolutional autoencoder on 36 steakhouse food images (6 of each of steak, potato, french fries, salads, burger, asparagus), and make similar image food recommendations based on the above algorithm to achieve a result of:

<p align="center"> 
<img src="https://github.com/ankonzoid/artificio/blob/master/similar_images_AE/output/result_burger_test.png" width="50%">
</p>

<p align="center"> 
<img src="https://github.com/ankonzoid/artificio/blob/master/similar_images_AE/output/result_salad_test.png" width="50%">
</p>

The model performs fairly well as a vanilla model with minimal fine-tuned training, in the sense that the top similar recommended images tend to be in same food category as the query image (i.e. querying a burger gives mostly burgers, and querying a salad gives mostly salads, ...). There is still much room for improvement in terms different neural network architectures, more/different training images, hyperparameter tuning to improve the generality of this model. 

The algorithm:

1) Train an autoencoder with training images in the same domain as the inventory images

2) Use the trained encoder to embed both the query images and the inventory images

3) Perform kNN (euclidean/cosine similarity) to find the inventory nearest neighbour image embeddings to the query image embeddings, and keep the k closest embeddings as the top-k recommendations

### Usage:

To make sure our similar images finder (trained on steakhouse food images) works on our test images, 

1. Run the command:
    
    > python similar_images_AE.py

When the run is complete, your answer images can be found in the `output` directory. However, if you would like to train the model from scratch then:
 
1. In `similar_images_AE.py`, set:
     
    * `model_name` to either `"simpleAE"` (1 FC hidden layer) or `"convAE"` (CNN)

    * `process_and_save_images = True` to perform the proper pre-processing of the images

    * `train_model = True` to instruct the program to train the model from scratch (it also saves it once the training is complete)

2. Run the command:

    > python similar_images_AE.py

### Required libraries:

* numpy, matplotlib, pylab, sklearn, keras, h5py, pillow

### Authors:

Anson Wong