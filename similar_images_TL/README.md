# Similar images finder (using transfer learning on VGG)

Given a set of database images, we take an pre-existing trained image classification model such as VGG, and apply transfer learning by removing its last layers and using the stripped model as a feature extractor on images. Doing so allows us to convert images into a set of feature vectors. 

The biggest benefit of transfer learning is that, unlike our household similar image autoencoder implementation, we only need to feed-forward our images into the model network (no training is needed at all).

<p align="center"> 
<img src="https://github.com/ankonzoid/artificio/blob/master/similar_images_TL/coverart/coverart.jpg" width="80%">
</p>



#### Most similar images to a query burger image:

<p align="center"> 
<img src="https://github.com/ankonzoid/artificio/blob/master/similar_images_TL/output/tsne.png" width="60%">
</p>

#### t-SNE of our small steakhouse food image database:

Although our image feature vectors by themselves are not easily interpretable by inspection, they can be mapped onto a 2-dimensional manifold via the t-SNE algorithm for visualization. 

<p align="center"> 
<img src="https://github.com/ankonzoid/artificio/blob/master/similar_images_TL/output/rec/burger_test_rec.png" width="80%">
</p>

### Usage:

The code can be run immediately by executing 

> python sim_img_TL.py 

The t-SNE embedding plot will be saved to the `output` directory. You can replace all the images in the `db` directory with your own images and simply run the code to obtain the t-SNE plot for your images.


### Required libraries:

* numpy, matplotlib, sklearn, keras, h5py, pillow