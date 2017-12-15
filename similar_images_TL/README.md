# Similar images finder (using transfer learning on VGG)

Given a set of database images, we take VGG (a trained image classification model), remove its last layers, and use it to convert our raw images into feature vectors. Note that no training is needed, we only need to feed-forward our images into our dissected VGG model.


<p align="center"> 
<img src="https://github.com/ankonzoid/artificio/blob/master/similar_images_TL/coverart/coverart.jpg" width="80%">
</p>

In this code, we query for similar steakhouse food images and can achieve a result of:

<p align="center"> 
<p float="left">
<img src="https://github.com/ankonzoid/artificio/blob/master/similar_images_TL/output/rec/burger_test_rec.png" width="60%">
<img src="https://github.com/ankonzoid/artificio/blob/master/similar_images_TL/output/rec/salad_test_rec.png" width="60%">
<img src="https://github.com/ankonzoid/artificio/blob/master/similar_images_TL/output/rec/asparagus_test_rec.png" width="60%">
</p>
</p>

#### t-SNE of our small steakhouse food image database:

Although our image feature vectors by themselves are not easily interpretable by inspection, they can be mapped onto a 2-dimensional manifold via the t-SNE algorithm for visualization. 

<p align="center"> 
<img src="https://github.com/ankonzoid/artificio/blob/master/similar_images_TL/output/tsne.png" width="50%">
</p>


### Usage:

The code can be run immediately by executing 

> python sim_img_TL.py 

The t-SNE embedding plot will be saved to the `output` directory. You can replace all the images in the `db` directory with your own images and simply run the code to obtain the t-SNE plot for your images.


### Required libraries:

* numpy, matplotlib, sklearn, keras, h5py, pillow