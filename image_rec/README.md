# Similar image recommendations

Given a set of query images (user) and a set of inventory images (store inventory), we find the top-k similar inventory images that are the most 'similar' to the set of query images using an encoder to encode the images as a vector and perform kNN to find 'similar' images. 

Explicitly we:

* 1) Train an autoencoder with training images in the same domain as the inventory images

* 2) Use the encoder to encode both the query images and the inventory images

* 3) Perform kNN (euclidean/cosine similarity) to find the nearest encoded inventory images to the encoded query images in the encoding space

* 4) Take the top-k closest encoding vectors as the top-k recommendations

Particularly for us, we use a convolutional autoencoder trained on 36 steakhouse food images (6 of each of regular steakhouse food items: steak, potato, french fries, salads, burger, asparagus), then make similar food recommendations based on the above algorithm. Below are results of querying test images of:


#### Query: a burger
<img src="https://github.com/ankonzoid/artificio/blob/master/image_rec/answer/result_burger.png" width="80%" align="center">

#### Query: a salad
<img src="https://github.com/ankonzoid/artificio/blob/master/image_rec/answer/result_salad.png" width="80%" align="center">

#### Query: asparagus
<img src="https://github.com/ankonzoid/artificio/blob/master/image_rec/answer/result_asparagus.png" width="80%" align="center">

The model performs fairly well as a vanilla model with minimal fine-tuned training, in the sense that the top similar recommended images tend to be in same food category as the query image (i.e. querying a burger gives mostly burgers, and querying a salad gives mostly salads, ...). There is still much room for improvement in terms different neural network architectures, more/different training images, hyperparameter tuning to improve the generality of this model. 

### Usage:

* 1) Place your query images into the `query` directory, inventory images into the `db/img_inventory_raw` directory, and training images into the `db/img_train_raw` directory

* 2) Run `python image_rec.py` from the `image_rec` directory

* 3) When the run is complete, your answer images will be deposited into the `answer` directory
