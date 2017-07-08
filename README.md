# artificio

A repository of deep learning codes which are tested and ready for real-world applications.

# Similar image recommendations
Given a set of query images (user given) and a set of inventory images (store inventory images), we find the top-k similar inventory images that are the most 'similar' to the set of query images. This can be achieved by training an autoencoder using a set of training images that are of the same domain as the inventory images, and using the encoder to cluster our images in the encoding space to introduce a measure of 'similarity' between the images.

For concreteness, we train a convolutional autoencoder using 36 training food images (6 of each of regular steakhouse food items: steak, potato, french fries, salads, burger, asparagus), then make recommendations based on encoded query image kNN of these items in encoding space. Below are results of querying unseen before images of:

### Query: a burger
<img src="https://github.com/ankonzoid/artificio/blob/master/image_rec/answer/result_burger.png" width="200%" align="center">

### Query: a salad
<img src="https://github.com/ankonzoid/artificio/blob/master/image_rec/answer/result_salad.png" width="200%" align="center">

### Query: asparagus
<img src="https://github.com/ankonzoid/artificio/blob/master/image_rec/answer/result_asparagus.png" width="200%" align="center">

The model performs fairly well for a vanilla model with minimal fine-tuned training, in the sense that the top similar recommended images tend to be in same food queried (querying burger gives mostly burgers, quering salads gives mostly salads, etc.). There are still different neural network architectures, more training images, hyperparameter tuning that can be done to improve the generality of this model. 


### Image autoencoder-knn method

1) Train an autoencoder with training images in hopefully the same domain as the inventory images
2) Use the encoder to encode both the query images and the inventory images
3) Perform an unsupervised clustering method (such as kNN) to find the nearest encoded inventory images to the encoded query images in the encoding space
4) Make the top-k recommendations using a specified similarity metric (such as distance or cosine similarity)

### Usage (relative to project root directory `artificio/image_rec`):
1) Place query images in the `query` directory, inventory images in the `db/img_inventory_raw` directory, and training images in the `db/img_train_raw` directory
2) Run `image_rec.py`
3) Your answer images will be created in the `answer` directory
