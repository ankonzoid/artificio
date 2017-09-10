# Similar image recommendations

Given a set of query images and a set of store inventory images, we find the top-k similar inventory images that are the most 'similar' to the set of query images in an unsupervised way of using an encoder to embed the images, then performing kNN in this embedding space to search for 'similar' images. For this code sample, we query for similar steakhouse food images:

<img src="https://github.com/ankonzoid/artificio/blob/master/image_rec/answer/result_burger.png" width="80%" align="center" caption="Steakhouse image recommendations when querying an image of a burger">

### Algorithm:

1) Train an autoencoder with training images in the same domain as the inventory images

2) Use the trained encoder to embed both the query images and the inventory images

3) Perform kNN (euclidean/cosine similarity) to find the inventory nearest neighbour image embeddings to the query image embeddings, and keep the k closest embeddings as the top-k recommendations

Particularly for our example code, we train a convolutional autoencoder on 36 steakhouse food images (6 of each of steak, potato, french fries, salads, burger, asparagus), and make similar image food recommendations based on the above algorithm. Below are more results of querying test images of salad and asparagus:

<img src="https://github.com/ankonzoid/artificio/blob/master/image_rec/answer/result_salad.png" width="80%" align="center">

<img src="https://github.com/ankonzoid/artificio/blob/master/image_rec/answer/result_asparagus.png" width="80%" align="center">

The model performs fairly well as a vanilla model with minimal fine-tuned training, in the sense that the top similar recommended images tend to be in same food category as the query image (i.e. querying a burger gives mostly burgers, and querying a salad gives mostly salads, ...). There is still much room for improvement in terms different neural network architectures, more/different training images, hyperparameter tuning to improve the generality of this model. 

### Usage:

The code can be run immediately by executing `python image_rec.py` using our pre-existing trained models. This run will output your query answer images into the `answer` directory. The full procedure for this is:

1) Place your query images into the `query` directory, inventory images into the `db/img_inventory_raw` directory, and training images into the `db/img_train_raw` directory. We already have default steakhouse food item images for you to use already.

2) Run `python image_rec.py` from the `image_rec` directory. When the run is complete, your answer images will be deposited into the `answer` directory.

If you would like to train the model from scratch, then open `image_rec.py` and:

* set `model_name` to either `"simpleAE"` (1 FC layer) or `"convAE"` (CNN)

* set `process_and_save_images = True` and use your own images

* set `train_model = True` to instruct code to train the model (and save it once it is trained)

### Required libraries:

* numpy, matplotlib, pylab, sklearn, keras, h5py, pillow