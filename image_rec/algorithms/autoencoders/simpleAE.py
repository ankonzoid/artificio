"""
 simpleAE.py  (author: Anson Wong / github: ankonzoid)

 Trains a simple autoencoder and saves its autoencoder, encoder, and decoder to file.
"""
# Set path for use on external servers
import os, sys, time, platform, datetime, pylab
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from ..utilities.IO import normalize_flatten_img_data

from ..utilities.IO import load_images_from_dir
from ..utilities.IO import load_images_from_dir_parallel
from ..utilities.IO import make_path
from ..utilities import naming_conventions

def main():
    project_root = os.path.dirname(__file__)
    sys.path.append(project_root)
    name_algo = "simpleAE"

    # =================================================
    # Set run parameters
    # =================================================
    training_dir = os.path.join(project_root, "bin", "")
    gray_scale = 0
    train_model = False  # train and save model?
    n_epochs = 20
    encode_dim = 128


    n_cores = 32
    parallelize = True  # use parallelization?
    use_mnist = False  # use mnist or custom data?
    n_clone = 1  # number of training image clones (look below to see how cloning is done)
    frac_training_use = 1  # fraction of training set to use
    ratio_training_test = 0.8  # training/test split ratio
    seed = 100  # training/test split seed
    optimizer = 'adam'
    loss = 'binary_crossentropy'

    # =================================================
    # Read and pre-process data
    # =================================================
    save_models = train_model # save the model?
    dir_dict = make_folder_filename_conventions(name_algo, training_heaven_dir, name_data, name_trainingset)
    if 0:
        for row in dir_dict:
            print(row, ":", dir_dict[row])
    training_dir = dir_dict["training_dir"]
    model_dir = dir_dict["model_dir"]
    model_subdir = dir_dict["model_subdir"]
    autoencoder_filename = dir_dict["autoencoder_filename"]
    autoencoder_checkpoint_filename = dir_dict["autoencoder_checkpoint_filename"]
    encoder_filename = dir_dict["encoder_filename"]
    decoder_filename = dir_dict["decoder_filename"]
    plot_filename_pdf = dir_dict["plot_filename_pdf"]
    plot_filename_png = dir_dict["plot_filename_png"]
    run_report_filename = dir_dict["run_report_filename"]
    embedding_matrix_filename = dir_dict["embedding_matrix_filename"]
    embedding_images_filename = dir_dict["embedding_images_filename"]

    autoencoder_arch_filename = dir_dict["autoencoder_arch_filename"]
    encoder_arch_filename = dir_dict["encoder_arch_filename"]
    decoder_arch_filename = dir_dict["decoder_arch_filename"]
    autoencoder_weights_filename = dir_dict["autoencoder_weights_filename"]
    encoder_weights_filename = dir_dict["encoder_weights_filename"]
    decoder_weights_filename = dir_dict["decoder_weights_filename"]

    make_path(model_dir)  # make general model path
    make_path(model_subdir)  # make specific model path

    print("Reading and normalizing training/test images...")
    if use_mnist:
        (x_train, _), (x_test, _) = mnist.load_data()
    elif train_model:
        # Assumes images are all of same size
        if parallelize:
            x_train, x_test, index_train, index_test, filenames_list = \
                load_images_from_dir_parallel(training_dir, ratio_training_test, frac_training_use,
                                              n_clone, seed, gray_scale, n_cores)
        else:
            x_train, x_test, index_train, index_test, filenames_list = \
                load_images_from_dir(training_dir, ratio_training_test, frac_training_use,
                                     n_clone, seed, gray_scale)
    else:  # custom random IO
        krandom_seed = 101
        frac_training_use = 0.01
        ratio_training_test = 0.2
        n_clone = 1
        x_train, x_test, index_train, index_test, filenames_list = \
            load_images_from_dir(training_dir, ratio_training_test, frac_training_use,
                                 n_clone, krandom_seed, gray_scale)

    ypixels = x_train.shape[1]
    xpixels = x_train.shape[2]
    if gray_scale:
        n_channels = 1
    else:
        n_channels = 3
    print(" x_train.shape = {0}".format(x_train.shape))
    print(" x_test.shape = {0}".format(x_test.shape))
    x_train = normalize_flatten_img_data(x_train)  # (n, total pixels)
    x_test = normalize_flatten_img_data(x_test)  # (n, total pixels)
    print(" x_train.shape (normalized) = {0}".format(x_train.shape))
    print(" x_test.shape (normalized) = {0}".format(x_test.shape))

    # =================================================
    # Train model (or load it)
    # =================================================
    if train_model:
        start_time = time.time()  # start timer

        # Start the report
        run_report = open(run_report_filename, "w")  # write report
        run_report.write(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
        run_report.write("\n\n")
        run_report.close()  # close report

        # Append to the report the computer and training specs
        run_report = open(run_report_filename, "a")  # append report
        run_report.write("--- full run start time = {0} ---\n".format(time.time() - start_time))
        run_report.write("\n")
        run_report.write(" platform: {0}\n".format(platform.machine()))
        run_report.write(" version: {0}\n".format(platform.version()))
        run_report.write(" system: {0}\n".format(platform.system()))
        run_report.write(" processor: {0}\n".format(platform.processor()))
        run_report.write("\n")
        run_report.write(" x_train.shape: {0}\n".format(x_train.shape))
        run_report.write(" x_test.shape: {0}\n".format(x_test.shape))
        run_report.write("\n")
        run_report.write(" training_heaven_dir: {0}\n".format(training_heaven_dir))
        run_report.write(" name_data: {0}\n".format(name_data))
        run_report.write(" name_trainingset: {0}\n".format(name_trainingset))
        run_report.write(" train_model: {0}\n".format(train_model))
        run_report.write(" use_mnist: {0}\n".format(use_mnist))
        run_report.write(" n_epochs: {0}\n".format(n_epochs))
        run_report.write(" n_clone: {0}\n".format(n_clone))
        run_report.write(" ratio_training_test: {0}\n".format(ratio_training_test))
        run_report.write(" seed: {0}\n".format(seed))
        run_report.write(" optimizer: {0}\n".format(optimizer))
        run_report.write(" loss: {0}\n".format(loss))
        run_report.write("\n")
        run_report.write(" training_dir: {0}\n".format(training_dir))
        run_report.write(" model_dir: {0}\n".format(model_dir))
        run_report.write(" model_subdir: {0}\n".format(model_subdir))
        run_report.write(" autoencoder_filename: {0}\n".format(autoencoder_filename))
        run_report.write(" autoencoder_checkpoint_filename: {0}\n".format(autoencoder_checkpoint_filename))
        run_report.write(" encoder_filename: {0}\n".format(encoder_filename))
        run_report.write(" decoder_filename: {0}\n".format(decoder_filename))
        run_report.write(" plot_filename_png: {0}\n".format(plot_filename_png))
        run_report.write(" plot_filename_pdf: {0}\n".format(plot_filename_pdf))
        run_report.close()  # close report

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create architecture
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        run_report = open(run_report_filename, "a")  # append report
        run_report.write("\n--- build model architecture start time = {0} ---\n\n".format(time.time() - start_time))
        run_report.close()  # close report
        if 1:
            # Create layers (x_train.shape = (n_train, flattened dim))
            input_img = Input(shape=(x_train.shape[1],))  # input layer
            encoded = Dense(encode_dim, activation='relu')(input_img)  # last encoded layer
            decoded = Dense(x_train.shape[1], activation='sigmoid')(encoded)  # last decoded layer

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set models
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set autoencoder model
        autoencoder = Model(input_img, decoded)
        print(autoencoder.summary())
        input_ae_shape = autoencoder.layers[0].input_shape[1:]
        output_ae_shape = autoencoder.layers[-1].output_shape[1:]

        # Set encoder model
        encoder = Model(input_img, encoded)  # set encoder
        input_enc_shape = encoder.layers[0].input_shape[1:]
        output_enc_shape = encoder.layers[-1].output_shape[1:]
        print(encoder.summary())

        # Set decoder model
        decoded_input = Input(shape=output_enc_shape)
        decoded_output = autoencoder.layers[-1](decoded_input)
        decoder = Model(decoded_input, decoded_output)
        input_dec_shape = decoder.layers[0].input_shape[1:]
        output_dec_shape = decoder.layers[-1].output_shape[1:]
        print(decoder.summary())

        # Print to report: architecture
        run_report = open(run_report_filename, "a")  # append report
        run_report.write("\n")
        run_report.write(" input_ae_shape: {0}\n".format(input_ae_shape))
        run_report.write(" output_ae_shape: {0}\n".format(output_ae_shape))
        for i in range(len(autoencoder.layers)):
            run_report.write(" autoencoder.layers[{0}]: input={1}, output={2}\n".format(i, autoencoder.layers[i].input_shape[1:],
                                                                                        autoencoder.layers[i].output_shape[1:]))
        run_report.write("\n")
        run_report.write(" input_enc_shape: {0}\n".format(input_enc_shape))
        run_report.write(" output_enc_shape: {0}\n".format(output_enc_shape))
        for i in range(len(encoder.layers)):
            run_report.write(" encoder.layers[{0}]: input={1}, output={2}\n".format(i, encoder.layers[i].input_shape[1:],
                                                                                    encoder.layers[i].output_shape[1:]))
        run_report.write("\n")
        run_report.write(" input_dec_shape: {0}\n".format(input_dec_shape))
        run_report.write(" output_dec_shape: {0}\n".format(output_dec_shape))
        for i in range(len(decoder.layers)):
            run_report.write(" decoder.layers[{0}]: input={1}, output={2}\n".format(i, decoder.layers[i].input_shape[1:],
                                                                                    decoder.layers[i].output_shape[1:]))
        run_report.close()  # close report

        # Compile
        autoencoder.compile(optimizer=optimizer, loss=loss)  # set loss and optimizer

        # Train model
        run_report = open(run_report_filename, "a")  # append report
        run_report.write("\n--- training start time = {0} ---\n\n".format(time.time() - start_time))
        run_report.close()  # close report

        # Model checkpoint, print when we have good success
        checkpoint = ModelCheckpoint(autoencoder_checkpoint_filename,
                                     monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        # Train model
        autoencoder.fit(x_train, x_train,
                        epochs = n_epochs, batch_size = 256,
                        callbacks = callbacks_list, verbose = 1,
                        shuffle = True,
                        validation_data = (x_test, x_test))

        # Print to report: models
        run_report = open(run_report_filename, "a")  # append report
        run_report.write("\n--- model saving time = {0} ---\n\n".format(time.time() - start_time))
        run_report.close()  # close report

        # Print finished run
        run_report = open(run_report_filename, "a")  # append report
        run_report.write("\n--- full run finished time = {0} ---\n\n".format(time.time() - start_time))
        run_report.close()  # close report
    else:
        autoencoder = load_model(autoencoder_filename)
        encoder = load_model(encoder_filename)
        decoder = load_model(decoder_filename)
        autoencoder.compile(optimizer=optimizer, loss=loss)  # set loss and optimizer
        encoder.compile(optimizer=optimizer, loss=loss)  # set loss and optimizer
        decoder.compile(optimizer=optimizer, loss=loss)  # set loss and optimizer
        print(autoencoder.summary())
        print(encoder.summary())
        print(decoder.summary())

    # Save model
    if save_models:
        autoencoder.save(autoencoder_filename)  # save full autoencoder model
        encoder.save(encoder_filename)  # save full encoder model
        decoder.save(decoder_filename)  # save full decoder model

        with open(autoencoder_arch_filename, "w+") as json_file:
            json_file.write(autoencoder.to_json())  # autoencoder arch: json format
        autoencoder.save_weights(autoencoder_weights_filename)  # autoencoder weights: hdf5 format
        with open(encoder_arch_filename, "w+") as json_file:
            json_file.write(encoder.to_json())  # encoder arch: json format
        encoder.save_weights(encoder_weights_filename)  # encoder weights: hdf5 format
        with open(decoder_arch_filename, "w+") as json_file:
            json_file.write(decoder.to_json())  # decoder arch: json format
        decoder.save_weights(decoder_weights_filename)  # decoder weights: hdf5 format

    # =================================================
    # Predict on our test set
    # =================================================
    encoded_imgs = encoder.predict(x_test)  # image encoded embedding
    print("encoded_imgs.shape=", encoded_imgs.shape)
    decoded_imgs = decoder.predict(encoded_imgs)  # image reconstruction
    print("decoded_imgs.shape=", decoded_imgs.shape)

    n = 10  # number of test images to take (test example indices 0, 1, ..., n-1)
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)  # display original
        if gray_scale:
            img_show_test = x_test[i].reshape((ypixels, xpixels))
            plt.imshow(img_show_test)
            plt.gray()
        else:
            img_show_test = x_test[i].reshape((ypixels, xpixels, n_channels))  # recombine to make RGB
            plt.imshow(img_show_test)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2,n,n+i+1)  # display reconstruction
        if gray_scale:
            decoded_img_show_test = decoded_imgs[i].reshape((ypixels, xpixels))
            plt.imshow(decoded_img_show_test)
            plt.gray()
        else:
            decoded_img_show_test = decoded_imgs[i].reshape((ypixels, xpixels, n_channels))  # recombine to make RGB
            plt.imshow(decoded_img_show_test)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    pylab.savefig(plot_filename_pdf)  # save plot
    pylab.savefig(plot_filename_png)  # save plot
    if 0:
        plt.show()

    # =================================================
    # Encode training set
    # =================================================
    if train_model:
        encoded_train_imgs = encoder.predict(x_train)  # image encoded embedding
        print("encoded_train_imgs.shape=", encoded_train_imgs.shape)
        np.save(embedding_matrix_filename, encoded_train_imgs)
        embimg_file = open(embedding_images_filename, 'w')
        for ind in index_train:
            embimg_file.write("{0}\n".format(filenames_list[ind]))


#
# Driver file
#
if __name__ == '__main__':
    main()
