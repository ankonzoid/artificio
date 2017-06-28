'''
 imageKNN.py (author: Anson Wong / github: ankonzoid)
 
 Image similarity recommender system using an autoencoder-clustering model.
 
 Autoencoder Method:
  1) Train an autoencoder (simple/Conv) on training images in 'db/images_training' 
  2) Saves trained autoencoder, encoder, and decoder to 'db/models'

 Clustering Method:
  3) Using our trained encoder in 'db/models', we encode inventory images in 'db/images_inventory'
  4) Train kNN model using encoded inventory images
  5) Encode query images in 'query', and predict their NN using our trained kNN model
  6) Compute a score for each inventory encoding relative to our query encoding (centroid/closest)
  7) Make k-recommendations by cloning top-k inventory images into 'answer'
'''
import sys, os
print("Python {0} on {1}".format(sys.version, sys.platform))
import shutil, glob
import numpy as np

from algorithms.utilities.image_manager import ImageManager
from algorithms.utilities.image_transformer import ImageTransformer
from algorithms.utilities.sorting import find_topk_unique
from algorithms.clustering.KNN import KNearestNeighbours
from algorithms.autoencoders import simpleAE
from algorithms.autoencoders import ConvAE

from keras.models import load_model

def main():
    project_root = os.path.dirname(__file__)
    sys.path.append(project_root)
    print("Project root: {0}".format(project_root))
    # ========================================
    # Set run settings
    # ========================================
    if 1:
        model_name = 'simpleAE'  # model folder
        flatten = True  # needs flattening for simpleAE encoder input
    elif 0:
        model_name = 'ConvAE'  # model folder
        flatten = False  # no need to flatten for ConvAE encoder input
    else:
        raise Exception("Invalid model name which is not simpleAE nor ConvAE")

    model_extension_tag = '_encoder.h5'  # encoder model h5 tag
    io_img_shape = (100, 100)  # force resize of raw images to (ypixels, xpixels)

    n_neighbors = 5  # number of nearest neighbours
    metric = "cosine"  # kNN metric (cosine only compatible with brute force)
    algorithm = "brute"  # search algorithm

    # Recommender mode:
    # 1 = nearest to centroid
    # 2 = nearest to any transaction point
    rec_mode = 2


    # ========================================
    # Generate expected file/folder paths and settings
    # ========================================
    # Assume project root directory to be directory of file
    project_root = os.path.dirname(__file__)

    # Query and answer folder
    query_dir = os.path.join(project_root, 'query')
    answer_dir = os.path.join(project_root, 'answer')

    # In database folder
    img_training_raw_dir = os.path.join(project_root, 'db/img_training_raw')
    img_inventory_raw_dir = os.path.join(project_root, 'db/img_inventory_raw')
    img_training_dir = os.path.join(project_root, 'db/img_training')
    img_inventory_dir = os.path.join(project_root, 'db/img_inventory')
    bin_training_dir = os.path.join(project_root, 'db/bin_training')
    bin_inventory_dir = os.path.join(project_root, 'db/bin_inventory')
    models_dir = os.path.join(project_root, 'db/models')

    # In algorithms folder
    autoencoders_dir = os.path.join(project_root, 'algorithms/autoencoders')
    clustering_dir = os.path.join(project_root, 'algorithms/clustering')
    IO_dir = os.path.join(project_root, 'algorithms/IO')
    utilities_dir = os.path.join(project_root, 'algorithms/utilities')

    encoder_filename = os.path.join(models_dir, model_name + model_extension_tag)

    # Set info file
    info = {
        "io_img_shape": io_img_shape,

        "query_dir": query_dir,
        "answer_dir": answer_dir,

        "img_training_raw_dir": img_training_raw_dir,
        "img_inventory_raw_dir": img_inventory_raw_dir,
        "img_training_dir": img_training_dir,
        "img_inventory_dir": img_inventory_dir,
        "bin_training_dir": bin_training_dir,
        "bin_inventory_dir": bin_inventory_dir,

        "autoencoders_dir": autoencoders_dir,
        "clustering_dir": clustering_dir,
        "IO_dir": IO_dir,
        "utilities_dir": utilities_dir,

        "encoder_filename": encoder_filename
    }

    exit()

    # ========================================
    #
    # Perform image processing
    #
    # ========================================

    # Initialize image transformer (and register encoder)
    print("Initializing ImageTransformer...")
    TR = ImageTransformer()  # provides functions for processing
    TR.configure(output_shape = info['output_shape'])

    # Initialize data manager (and register encoder)
    print("Initializing DataManager...")
    DM = ImageManager()  # provides functions for IO
    DM.configure(info)



    exit()

    # ========================================
    #
    # Train autoencoder
    #
    # ========================================
    train_autoencoder = True
    if train_autoencoder:
        pass

    # ========================================
    #
    # Perform clustering recommendation
    #
    # ========================================

    # Load encoder
    encoder = load_model(encoder_filename)
    encoder.compile(optimizer='adam', loss='binary_crossentropy')  # set loss and optimizer


    # Read raw image data, then forced resize them
    print("Load raw image data and resize them...")
    x_train_raw = DM.load_raw_data(batch_size = 5000)

    print("x_train_raw.shape = {0}".format(x_train_raw.shape))
    DM.build_mapping()


    # Encode raw data, then flatten the encoding dimensions
    print("Encoding resized raw image data and flatten encoding dimensions...")
    x_train_enc_flatten = DM.encode_raw(x_train_raw)  # takes raw data input, outputs flattened encoding
    print("x_train_enc_flatten.shape = {0}".format(x_train_enc_flatten.shape))

    # =================================
    # Train kNN model
    # =================================
    print("Performing kNN to locate nearby items to user centroid points...")
    EMB = KNearestNeighbours()  # initialize embedding kNN class
    EMB.compile(n_neighbors = n_neighbors, algorithm = algorithm, metric = metric)  # compile kNN model
    EMB.fit(x_train_enc_flatten)  # fit kNN

    # =================================
    # Perform kNN to the centroid query point
    # =================================
    while True:
        # Read items in query folder
        print("Reading query images from query folder: {0}".format(query_folder))
        for j, batch in enumerate(TR.transform_all(query_folder, grey_scale = False)):
            x_query_raw = batch

            # Encode all raw query images in query folder
            print("[batch {0}]".format(j))
            print("x_query_raw.shape = {0}".format(x_query_raw.shape))
            x_query_enc_flatten = DM.encode_raw(x_query_raw)


            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute distances and indices for recommendation
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if rec_mode == 1:  # kNN centroid transactions
                # Compute centroid point of the query encoding vectors (equal weights)
                x_centroid_enc_flatten = np.mean(x_query_enc_flatten, axis = 0)
                print("x_centroid_enc_flatten.shape = {0}".format(x_centroid_enc_flatten.shape))
                # Find nearest neighbours to centroid point
                distances, indices = EMB.predict(np.array([x_centroid_enc_flatten]))

            elif rec_mode == 2:  # kNN all transactions
                # Find k nearest neighbours to all transactions, then flatten the distances and indices
                distances, indices = EMB.predict(x_query_enc_flatten)
                distances = distances.flatten()
                indices = indices.flatten()
                # Pick k unique training indices which have the shortest distances any transaction point
                indices, distances = find_topk_unique(indices, distances, n_neighbors)

            else:
                raise Exception("Invalid method for making recommendations")
            print("distances.shape = {0}".format(distances.shape))
            print("indices.shape = {0}".format(indices.shape))


            # Make k-recommendations and output images and ids into the answer folder
            #
            # Backward-map indices -> vectors, and check euclidean distance scores.
            # If backward-mapping is correct, the k=1 NN distance is zero for a training image query
            print("Copying the k-recommendeded training images to answer folder '{0}'...".format(answer_folder))
            for i, (index, distance) in enumerate(zip(indices, distances)):
                print("({0}.{1}): indices = {2}".format(j, i, index))
                print("({0}.{1}): score = {2}".format(j, i, distance))
                answer_file_list = [DM.get_file_name(x) for x in index]
                print(answer_file_list)

                # Go through the answer filenames, and clone recommended training images to answer folder
                print("Cloning k-recommended raw images to answer folder '{0}'".format(
                    answer_folder))
                itemid_rec = []
                internalid_rec = []
                for k_rec, answer_file in enumerate(answer_file_list):

                    # Extract answer filename
                    itemid_k_str, tag = DM.extract_name_tag_filename(answer_file)  # filename with real itemid
                    internalid_k = DM.get_index(answer_file)
                    #answer_filename = os.path.join(answer_folder, str(internalid_k)+'.jpg')  # filename with internal id
                    answer_filename = os.path.join(answer_folder, itemid_k_str+'.jpg')  # filename with item id

                    # Append to itemid/internalid receommendation list
                    itemid_rec.append(int(itemid_k_str))
                    internalid_rec.append(int(internalid_k))

                    # Clone answer file to answer folder
                    shutil.copy(answer_file, answer_filename)

                # Print itemid/internalid recommendations to console
                print("itemid_rec = {0}".format(itemid_rec))
                print("internalid_rec = {0}".format(internalid_rec))

                # Print itemid/internalid recommendations to file
                print("Printing k-recommended itemids/internalids to text in answer folder '{0}'".format(answer_folder))
                itemid_internalid_rec_filename = os.path.join(answer_folder, 'itemid_internalid_rec.txt')
                with open(itemid_internalid_rec_filename, "w") as fid:
                    fid.write("itemid, internalid\n")
                    for (itemid_i, internalid_i) in zip(itemid_rec, internalid_rec):
                        fid.write("%d, %d\n" % (itemid_i, internalid_i))

        # Wait for input (used for predicting other queries)
        c = input('Continue? Type `q` to break\n')
        if c == 'q':
            break

# ==========================
# Side functions
# ==========================
def check_unique_itemids(img_training_raw_dir, IDM):
    # Make sure itemids are all unique
    if check_unique_itemids:
        # Collect itemids from their names
        itemids_check = np.empty(shape=(0), dtype=int)
        for path in [img_training_raw_dir]:
            filenames_list = glob.glob(path + "/*")

            # Get itemids from this path
            itemids_path = np.zeros((len(filenames_list)), dtype=int)
            for i, filename_i in enumerate(filenames_list):
                name, tag = IDM.extract_name_tag_filename(filename_i)
                #print("i = {0}, name = {1}, tag = {2}".format(i, name, tag))
                itemids_path[i] = int(name)

            # Append these itemids to global itemids
            itemids_check = np.append(itemids_check, itemids_path)

        # Make checks that there are no duplicate itemids
        n_itemids = len(itemids_check)
        n_itemids_unique = len(np.unique(itemids_check))
        if n_itemids != n_itemids_unique:
            raise Exception("The set of global itemids has duplicate itemids!")



# Driver
if __name__ == "__main__":
    main()
