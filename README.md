# artificio

A repository of deep learning codes which are tested and ready for real-world applications.

# Similar image recommendations (for stores)
Given a set of query images (user given) and a set of inventory images (store product images), we find the top-k similar inventory images that are most 'similar' to the set of query images. The method we use is:

### 1) Create/extract query images
<img src="https://github.com/ankonzoid/artificio/blob/master/query/result_hamburger.png" width="200%" align="center">

<img src="https://github.com/ankonzoid/artificio/blob/master/query/result_salad.png" width="200%" align="center">

<img src="https://github.com/ankonzoid/artificio/blob/master/query/result_asparagus.png" width="200%" align="center">

1) Train an autoencoder with training images in hopefully the same domain as the inventory images
2) Use the encoder to encode both the query images and the inventory images
3) Perform an unsupervised clustering method (such as kNN) to find the nearest encoded inventory images to the encoded query images in the encoding space
4) Make the top-k recommendations using a specified similarity metric (such as distance or cosine similarity)

### Usage (relative to project root directory `artificio/image_rec`):
1) Place query images in the `query` directory, inventory images in the `db/img_inventory_raw` directory, and training images in the `db/img_train_raw` directory
2) Run `image_rec.py`
3) Your answer images will be created in the `answer` directory
