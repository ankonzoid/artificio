'''
 image_rec.py (author: Anson Wong / git: ankonzoid)
 
 Image similarity recommender system using an autoencoder-clustering model.
 
 Autoencoder method:
  1) Train an autoencoder (simple/Conv) on training images in 'db/images_training' 
  2) Saves trained autoencoder, encoder, and decoder to 'db/models'

 Clustering method:
  3) Using our trained encoder in 'db/models', we encode inventory images in 'db/images_inventory'
  4) Train kNN model using encoded inventory images
  5) Encode query images in 'query', and predict their NN using our trained kNN model
  6) Compute a score for each inventory encoding relative to our query encoding (centroid/closest)
  7) Make k-recommendations by cloning top-k inventory images into 'answer'
'''
import sys, os, shutil
print("Python {0} on {1}".format(sys.version, sys.platform))
import numpy as np

from algo.autoencoders.AE import AE
from algo.clustering.KNN import KNearestNeighbours
from algo.utilities.image_utilities import ImageUtils
from algo.utilities.sorting import find_topk_unique

def main():
    # ========================================
    # Set run settings
    # ========================================
    #################################
    #################################
    if 0:
        model_name = 'simpleAE'
        flatten_before_encode = True
        flatten_after_encode = False
    elif 1:
        model_name = 'convAE'
        flatten_before_encode = False
        flatten_after_encode = True
    else:
        raise Exception("Invalid model name which is not simpleAE/convAE")

    ############################
    ###   Image processing   ###
    ############################
    process_and_save_images = False

    ###########################################
    ###   Autoencoder training parameters   ###
    ###########################################
    train_autoencoder = False
    img_shape = (100, 100)  # force resize -> (ypixels, xpixels)
    ratio_train_test = 0.8
    seed = 100

    loss = "binary_crossentropy"
    optimizer = "adam"
    n_epochs = 100
    batch_size = 256

    save_reconstruction_on_load_model = True

    ###################################
    ###   KNN training parameters   ###
    ###################################
    n_neighbors = 5  # number of nearest neighbours
    metric = "cosine"  # kNN metric (cosine only compatible with brute force)
    algorithm = "brute"  # search algorithm
    recommendation_method = 2  # 1 = centroid kNN, 2 = all points kNN



    # ========================================
    # Generate expected file/folder paths and settings
    # ========================================
    # Assume project root directory to be directory of file
    project_root = os.path.dirname(__file__)
    print("Project root: {0}".format(project_root))

    # Query and answer folder
    query_dir = os.path.join(project_root, 'query')
    answer_dir = os.path.join(project_root, 'answer')

    # In database folder
    db_dir = os.path.join(project_root, 'db')
    img_train_raw_dir = os.path.join(db_dir, 'img_train_raw')
    img_inventory_raw_dir = os.path.join(db_dir, 'img_inventory_raw')
    img_train_dir = os.path.join(db_dir, 'img_train')
    img_inventory_dir = os.path.join(db_dir, 'img_inventory')

    # Run output
    models_dir = os.path.join(db_dir, 'models')

    # Set info file
    info = {
        # Run settings
        "img_shape": img_shape,
        "flatten_before_encode": flatten_before_encode,
        "flatten_after_encode": flatten_after_encode,

        # Directories
        "query_dir": query_dir,
        "answer_dir": answer_dir,

        "img_train_raw_dir": img_train_raw_dir,
        "img_inventory_raw_dir": img_inventory_raw_dir,
        "img_train_dir": img_train_dir,
        "img_inventory_dir": img_inventory_dir,

        # Run output
        "models_dir": models_dir
    }

    # Initialize image utilities (and register encoder)
    IU = ImageUtils()
    IU.configure(info)

    # ========================================
    #
    # Pre-process save/load training and inventory images
    #
    # ========================================

    # Process and save
    if process_and_save_images:

        # Training images
        IU.raw2resized_load_save(raw_dir=img_train_raw_dir,
                                 processed_dir=img_train_dir,
                                 img_shape=img_shape)
        # Inventory images
        IU.raw2resized_load_save(raw_dir=img_inventory_raw_dir,
                                 processed_dir=img_inventory_dir,
                                 img_shape=img_shape)


    # ========================================
    #
    # Train autoencoder
    #
    # ========================================

    # Set up autoencoder base class
    MODEL = AE()

    MODEL.configure(model_name=model_name)

    if train_autoencoder:

        print("Training the autoencoder...")

        # Generate naming conventions
        dictfn = MODEL.generate_naming_conventions(model_name, models_dir)
        MODEL.start_report(dictfn)  # start report

        # Load training images to memory (resizes when necessary)
        x_data_all, all_filenames = \
            IU.raw2resizednorm_load(raw_dir=img_train_dir, img_shape=img_shape)
        print("\nAll data:")
        print(" x_data_all.shape = {0}\n".format(x_data_all.shape))

        # Split images to training and validation set
        x_data_train, x_data_test, index_train, index_test = \
            IU.split_train_test(x_data_all, ratio_train_test, seed)
        print("\nSplit data:")
        print("x_data_train.shape = {0}".format(x_data_train.shape))
        print("x_data_test.shape = {0}\n".format(x_data_test.shape))

        # Flatten data if necessary
        if flatten_before_encode:
            x_data_train = IU.flatten_img_data(x_data_train)
            x_data_test = IU.flatten_img_data(x_data_test)
            print("\nFlattened data:")
            print("x_data_train.shape = {0}".format(x_data_train.shape))
            print("x_data_test.shape = {0}\n".format(x_data_test.shape))

        # Set up architecture and compile model
        MODEL.set_arch(input_shape=x_data_train.shape[1:],
                       output_shape=x_data_train.shape[1:])
        MODEL.compile(loss=loss, optimizer=optimizer)
        MODEL.append_arch_report(dictfn)  # append to report

        # Train model
        MODEL.append_message_report(dictfn, "Start training")  # append to report
        MODEL.train(x_data_train, x_data_test,
                    n_epochs=n_epochs, batch_size=batch_size)
        MODEL.append_message_report(dictfn, "End training")  # append to report

        # Save model to file
        MODEL.save_model(dictfn)

        # Save reconstructions to file
        MODEL.plot_save_reconstruction(x_data_test, img_shape, dictfn, n_plot=10)

    else:

        # Generate naming conventions
        dictfn = MODEL.generate_naming_conventions(model_name, models_dir)

        # Load models
        MODEL.load_model(dictfn)

        # Compile model
        MODEL.compile(loss=loss, optimizer=optimizer)

        # Save reconstructions to file
        if save_reconstruction_on_load_model:
            x_data_all, all_filenames = \
                IU.raw2resizednorm_load(raw_dir=img_train_dir, img_shape=img_shape)
            if flatten_before_encode:
                x_data_all = IU.flatten_img_data(x_data_all)
            MODEL.plot_save_reconstruction(x_data_all, img_shape, dictfn, n_plot=10)

    # ========================================
    #
    # Perform clustering recommendation
    #
    # ========================================

    # Load inventory images to memory (resizes when necessary)
    x_data_inventory, inventory_filenames = \
        IU.raw2resizednorm_load(raw_dir=img_inventory_dir, img_shape=img_shape)
    print("\nx_data_inventory.shape = {0}\n".format(x_data_inventory.shape))

    # Explictly assign loaded encoder
    encoder = MODEL.encoder

    # Encode our data, then flatten to encoding dimensions
    # We switch names for simplicity: inventory -> train, query -> test
    print("Encoding data and flatten its encoding dimensions...")
    if flatten_before_encode:  # Flatten the data before encoder prediction
        x_data_inventory = IU.flatten_img_data(x_data_inventory)

    x_train_kNN = encoder.predict(x_data_inventory)

    if flatten_after_encode:  # Flatten the data after encoder prediction
        x_train_kNN = IU.flatten_img_data(x_train_kNN)

    print("\nx_train_kNN.shape = {0}\n".format(x_train_kNN.shape))


    # =================================
    # Train kNN model
    # =================================
    print("Performing kNN to locate nearby items to user centroid points...")
    EMB = KNearestNeighbours()  # initialize embedding kNN class
    EMB.compile(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)  # compile kNN model
    EMB.fit(x_train_kNN)  # fit kNN


    # =================================
    # Perform kNN to the centroid query point
    # =================================
    while True:

        # Read items in query folder
        print("Reading query images from query folder: {0}".format(query_dir))

        # Load query images to memory (resizes when necessary)
        x_data_query, query_filenames = \
            IU.raw2resizednorm_load(raw_dir=query_dir,
                                    img_shape=img_shape)
        print("\nx_data_query.shape = {0}\n".format(x_data_query.shape))

        # Encode query images
        if flatten_before_encode:  # Flatten the data before encoder prediction
            x_data_query = IU.flatten_img_data(x_data_query)

        x_test_kNN = encoder.predict(x_data_query)

        if flatten_after_encode:  # Flatten the data after encoder prediction
            x_test_kNN = IU.flatten_img_data(x_test_kNN)

        print("\nx_test_kNN.shape = {0}\n".format(x_test_kNN.shape))


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute distances and indices for recommendation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if recommendation_method == 1:  # kNN centroid transactions
            # Compute centroid point of the query encoding vectors (equal weights)
            x_test_kNN_centroid = np.mean(x_test_kNN, axis = 0)
            # Find nearest neighbours to centroid point
            distances, indices = EMB.predict(np.array([x_test_kNN_centroid]))

        elif recommendation_method == 2:  # kNN all transactions
            # Find k nearest neighbours to all transactions, then flatten the distances and indices
            distances, indices = EMB.predict(x_test_kNN)
            distances = distances.flatten()
            indices = indices.flatten()
            # Pick k unique training indices which have the shortest distances any transaction point
            indices, distances = find_topk_unique(indices, distances, n_neighbors)

        else:
            raise Exception("Invalid method for making recommendations")

        print("\ndistances.shape = {0}".format(distances.shape))
        print("indices.shape = {0}\n".format(indices.shape))

        # Make k-recommendations and clone most similar inventory images to answer dir
        print("Cloning k-recommended inventory images to answer folder '{0}'...".format(answer_dir))
        for i, (index, distance) in enumerate(zip(indices, distances)):
            print("\n({0}): index = {1}".format(i, index))
            print("({0}): distance = {1}\n".format(i, distance))

            for k_rec, ind in enumerate(index):

                # Extract inventory filename
                inventory_filename = inventory_filenames[ind]

                # Extract answer filename
                name, tag = IU.extract_name_tag(inventory_filename)
                answer_filename = os.path.join(answer_dir, name + '.' + tag)

                # Clone answer file to answer folder
                print("Cloning '{0}' to answer directory...".format(inventory_filename))
                shutil.copy(inventory_filename, answer_filename)

        # Wait for input (used for predicting other queries)
        if 1:
            break
        else:
            c = input('Continue? Type `q` to break\n')
            if c == 'q':
                break


# Driver
if __name__ == "__main__":
    main()
