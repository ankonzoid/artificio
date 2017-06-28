# artificio

A repository of deep learning codes which are tested and ready for real-world applications.

## Similar item recommendations (for stores)
Given a set of query images (user given) and a set of inventory images (store product images), we find the top-k similar inventory images that are most 'similar' to the set of query images. The method we use is:

1) Train an autoencoder with training images in hopefully the same domain as the inventory images
2) Use the encoder to encode both the query images and the inventory images
3) Perform an unsupervised clustering method (such as kNN) to find the nearest encoded inventory images to the encoded query images in the encoding space
4) Make the top-k recommendations using a specified similarity metric (such as distance or cosine similarity)
