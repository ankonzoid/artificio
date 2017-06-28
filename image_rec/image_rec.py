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
import os, shutil, glob
import numpy as np

from algorithms.utilities.image_manager import ImageManager
from algorithms.utilities.image_transformer import ImageTransformer
from algorithms.utilities.sorting import find_topk_unique
from algorithms.clustering.KNN import KNearestNeighbours

from algorithms.autoencoder import simpleAE
from algorithms.autoencoder import ConvAE

from keras.models import load_model

def main():
    # ========================================
    # Set run settings
    # ========================================
    company_name = 'plenty'  # company db name in db_folder
    img_type_name = 'raw'  # use raw images files (we have image transformer)
    if 1:
        model_name = 'simpleAE_rgb_resized'  # model folder
        flatten = True  # needs flattening for simpleAE encoder input
    elif 0:
        model_name = 'ConvAE_rgb_resized'  # model folder
        flatten = False  # no need to flatten for ConvAE encoder input
    else:
        raise Exception("Invalid model name which is not simpleAE nor ConvAE")

    model_extension_tag = '_encoder.h5'  # encoder model h5 tag
    output_shape = (100, 100)  # force resize of raw images
    grey_scale = False

    n_neighbors = 5  # number of nearest neighbours
    metric = "cosine"  # kNN metric (cosine only compatible with brute force)
    algorithm = "brute"  # search algorithm

    # Recommender mode:
    # 1 = nearest to centroid
    # 2 = nearest to any transaction point
    rec_mode = 2

    # ========================================
    #
    # Train autoencoder
    #
    # ========================================
    train_autoencoder = True
    if train_autoencoder:


    # ========================================
    #
    # Perform clustering recommendation
    #
    # ========================================
    load_encoder = True
    if load_encoder:





    # ========================================
    # Generate expected file/folder paths and settings
    # ========================================
    cur_dir = os.path.dirname(__file__)  # supposedly imageKNN and KNN folder
    project_root = os.path.join(cur_dir, '..', '..')  # rex_framework directory (project root directory)
    prototype_folder = os.path.join(project_root, 'prototype')

    app_folder = os.path.join(prototype_folder, 'image_knn')  # imageKNN and KNN folder
    db_folder = os.path.join(prototype_folder, 'db')  # database folder of kNN training data
    query_folder = os.path.join(prototype_folder, 'query')  # folder of query images
    answer_folder = os.path.join(prototype_folder, 'answer')  # folder of answer images

    db_company_folder = os.path.join(db_folder, company_name)  # e.g. plenty
    train_db_folder = os.path.join(db_company_folder, 'train', img_type_name)  # training data for kNN (inventory images)
    model_db_folder = os.path.join(db_company_folder, 'models', model_name)

    train_db_paths = [train_db_folder]  # list of raw kNN training images
    train_bin_db_paths = [train_bin_db_folder]  # list of binary kNN training images
    encoder_filename = os.path.join(model_db_folder, model_name + model_extension_tag)

    # Set config file to be used for the remainder of code
    config = {
        'db_name': company_name,
        'grey_scale': grey_scale,
        'flatten': flatten,
        'output_shape': output_shape,
        'query_folder': query_folder,
        'answer_folder': answer_folder,
        'train_db_paths': train_db_paths,
        'train_bin_db_paths': train_bin_db_paths,
        'encoder_filename': encoder_filename
    }

    # Make tests before proceeding
    if len(train_db_paths) != len(train_bin_db_paths):
        raise Exception("len(train_db_paths) != len(train_bin_db_paths)")

    # Initialize encoder
    encoder = load_model(encoder_filename)
    encoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')  # set loss and optimizer

    # Initialize image transformer (and register encoder)
    print("Initializing ImageTransformer...")
    t = ImageTransformer()
    t.configure(output_shape = config['output_shape'])
    t.register_encoder(encoder)

    # Initialize data manager (and register encoder)
    print("Initializing DataManager...")
    dm = DataManager()
    dm.configure(config)
    dm.register_encoder(encoder)

    # Read raw image data, then forced resize them
    print("Load raw image data and resize them...")
    x_train_raw = dm.load_raw_data(batch_size = 5000)  # resizes -> (n_train, y_img, x_img, n_channels_img)
    print("x_train_raw.shape = {0}".format(x_train_raw.shape))
    dm.build_mapping()

    # Make sure itemids are all unique
    check_unique_itemids = True
    if check_unique_itemids:
        # Collect itemids from their names
        itemids_check = np.empty(shape=(0), dtype=int)
        for path in train_db_paths:
            filenames_list = glob.glob(path + "/*")

            # Get itemids from this path
            itemids_path = np.zeros((len(filenames_list)), dtype=int)
            for i, filename_i in enumerate(filenames_list):
                name, tag = dm.extract_name_tag_filename(filename_i)
                #print("i = {0}, name = {1}, tag = {2}".format(i, name, tag))
                itemids_path[i] = int(name)

            # Append these itemids to global itemids
            itemids_check = np.append(itemids_check, itemids_path)

        # Make checks that there are no duplicate itemids
        n_itemids = len(itemids_check)
        n_itemids_unique = len(np.unique(itemids_check))
        if n_itemids != n_itemids_unique:
            raise Exception("The set of global itemids has duplicate itemids!")


    # Encode raw data, then flatten the encoding dimensions
    print("Encoding resized raw image data and flatten encoding dimensions...")
    x_train_enc_flatten = dm.encode_raw(x_train_raw)  # takes raw data input, outputs flattened encoding
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
        for j, batch in enumerate(t.transform_all(query_folder, grey_scale = grey_scale)):
            x_query_raw = batch

            # Encode all raw query images in query folder
            print("[batch {0}]".format(j))
            print("x_query_raw.shape = {0}".format(x_query_raw.shape))
            x_query_enc_flatten = dm.encode_raw(x_query_raw)


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
                answer_file_list = [dm.get_file_name(x) for x in index]
                print(answer_file_list)

                # Go through the answer filenames, and clone recommended training images to answer folder
                print("Cloning k-recommended raw images to answer folder '{0}'".format(
                    answer_folder))
                itemid_rec = []
                internalid_rec = []
                for k_rec, answer_file in enumerate(answer_file_list):

                    # Extract answer filename
                    itemid_k_str, tag = dm.extract_name_tag_filename(answer_file)  # filename with real itemid
                    internalid_k = dm.get_index(answer_file)
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

# Driver
if __name__ == "__main__":
    main()
