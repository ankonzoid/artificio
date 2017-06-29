"""
 simpleAE.py  (author: Anson Wong / github: ankonzoid)

 Simple autoencoder class
"""
from .embedding_base import EmbeddingBase

import datetime, os, platform, pylab, sys, time
import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model

class SimpleAE(EmbeddingBase):

    def __init__(self):

        # For model architecture and weights (used for load/save)
        self.autoencoder = None
        self.autoencoder_input_shape = None
        self.autoencoder_output_shape = None

        self.encoder = None
        self.encoder_input_shape = None
        self.encoder_output_shape = None

        self.decoder = None
        self.decoder_input_shape = None
        self.decoder_output_shape = None

        # Some sanity parameters to be used in the middle of the run
        self.is_compiled = None

        # From custom information json file
        self.training_image_shape = None

        # User-parameters for run
        self.name = None
        self.train_model = None

        # User-parameters for task data set
        self.task_data_filename = None

        # User-parameters for training/test data set
        self.ypixels = None
        self.xpixels = None

        # User-parameters for architecture
        self.n_epochs = None
        self.batch_size = None
        self.optimizer = None
        self.loss = None

        super().__init__()

    """
     Configure
    """
    def configure(self, info):



        # Available user-parameters for architecture
        self.loss = info["loss"]  # pick model loss function
        self.optimizer = info["optimizer"]  # pick model optimizer
        self.n_epochs = info["n_epochs"]  # number of training epochs
        self.batch_size = info["batch_size"]  # training batch size

    """
     Check info file
    """
    def check_info(self, info):
        for param in info:
            print(param)

    ### ====================================================================
    ###
    ### Training
    ###
    ### ====================================================================

    """
     Train the autoencoder
    """
    def fit(self, x_train, x_test):
        self.autoencoder.fit(x_train, x_train,
                             epochs = self.n_epochs,
                             batch_size = self.batch_size,
                             shuffle = True,
                             validation_data = (x_test, x_test))
    
    ### ====================================================================
    ###
    ### Testing
    ###
    ### ====================================================================

    """
     Encode the image (using trained encoder)
    """
    def encode(self, img):
        if not self.is_compiled:
            raise Exception("Not compiled yet!")
        x_encoded = self.encoder.predict(img)
        return x_encoded

    """
     Decode the encoded image (using trained decoder)
    """
    def decode(self, x_encoded):
        if not self.is_compiled:
            raise Exception("Not compiled yet!")
        img_decoded = self.decoder.predict(x_encoded)
        return img_decoded

    ### ====================================================================
    ###
    ### Data IO
    ###
    ### ====================================================================

    """
     Set the training data
    """
    def set_training_data(self, userparam_filename):
        # =================================================
        # Read user parameter file
        # =================================================
        with open(userparam_filename, 'r') as json_file:
            userparam = json.load(json_file)

        self.use_existing_training_resized = userparam["training_data_parameters"]["use_existing_resized"]
        self.ratio_of_training_data_used = userparam["training_data_parameters"]["ratio_used"]
        self.training_data_dir = userparam["training_data_parameters"]["data_dir"]
        self.training_data_resized_dir = userparam["training_data_parameters"]["data_resized_dir"]
        self.ypixels_force_resize = userparam["training_data_parameters"]["ypixels_force_resize"]
        self.xpixels_force_resize = userparam["training_data_parameters"]["xpixels_force_resize"]
        self.image_tag = userparam["training_data_parameters"]["image_tag"]

    """
     Set the testing data
    """
    def set_test_data(self, info):
        # =================================================
        # Read user parameter file
        # =================================================
        self.task_data_filename = info["task_data_parameters"]["data_filename"]



    ### ====================================================================
    ###
    ### Model IO
    ###
    ### ====================================================================

    """
     Load model architecture and weights of autoencoder, encoder, and decoded
    """
    def load(self, autoencoder_filename, encoder_filename, decoder_filename, info):

        # Load custom information
        self.training_image_shape = info['training_image_shape']  # training image shape (n, ypixels, xpixels)

        # Load autoencoder architecture + weights + shapes
        self.autoencoder = load_model(autoencoder_filename)
        self.autoencoder_input_shape = self.autoencoder.input_shape  # set input shape from loaded model
        self.autoencoder_output_shape = self.autoencoder.output_shape  # set output shape from loaded model

        # Load encoder architecture + weights + shapes
        self.encoder = load_model(encoder_filename)
        self.encoder_input_shape = self.encoder.input_shape  # set input shape from loaded model
        self.encoder_output_shape = self.encoder.output_shape  # set output shape from loaded model

        # Load decoder architecture + weights + shapes
        self.decoder = load_model(decoder_filename)
        self.decoder_input_shape = self.decoder.input_shape  # set input shape from loaded model
        self.decoder_output_shape = self.decoder.output_shape  # set output shape from loaded model

    """
     Save model architecture and weights of autoencoder, encoder, and decoder
    """
    def save(self, autoencoder_filename, encoder_filename, decoder_filename):
        self.autoencoder.save(autoencoder_filename)
        self.encoder.save(encoder_filename)
        self.decoder.save(decoder_filename)


    ### ====================================================================
    ###
    ### Architecture compilation
    ###
    ### ====================================================================

    """
     Set NN architecture
    """
    def arch(self, info):

            # =================================================
            # Set training parameters
            # =================================================

            # =================================================
            # Create neural network architecture
            # =================================================
            input_shape =
            output_shape =


            # =================================
            # Create hidden layers
            # =================================

            # Encoding hidden layers
            input_img = Input(shape=(x_train.shape[1],))
            encoded = Dense(encode_dim, activation='relu')(input_img)

            # Decoding hidden layers
            decoded = Dense(x_train.shape[1], activation='sigmoid')(encoded)

            # =================================
            # Create models
            # =================================

            # ~~~~~~~~~~~~~~~~~~~
            # Create autoencoder model
            # ~~~~~~~~~~~~~~~~~~~
            autoencoder = Model(input_img, decoded)
            print(autoencoder.summary())
            # Set encoder input/output dimensions
            input_autoencoder_shape = autoencoder.layers[0].input_shape[1:]
            output_autoencoder_shape = autoencoder.layers[-1].output_shape[1:]

            # ~~~~~~~~~~~~~~~~~~~
            # Create encoder model
            # ~~~~~~~~~~~~~~~~~~~
            encoder = Model(input_img, encoded)  # set encoder
            print(encoder.summary())
            # Set encoder input/output dimensions
            input_encoder_shape = encoder.layers[0].input_shape[1:]
            output_encoder_shape = encoder.layers[-1].output_shape[1:]

            # ~~~~~~~~~~~~~~~~~~~
            # Create decoder model
            # ~~~~~~~~~~~~~~~~~~~
            decoded_input = Input(shape=output_encoder_shape)
            decoded_output = autoencoder.layers[-1](decoded_input)
            decoder = Model(decoded_input, decoded_output)
            print(decoder.summary())
            # Set encoder input/output dimensions
            input_decoder_shape = decoder.layers[0].input_shape[1:]
            output_decoder_shape = decoder.layers[-1].output_shape[1:]

            # =================================================
            # Set models
            # =================================================
            self.autoencoder = autoencoder  # untrained
            self.encoder = encoder  # untrained
            self.decoder = decoder  # untrained

            # =================================================
            # Set model input/output dimensions
            # We provide two options for getting these numbers:
            # - via the individual layers
            # - via the models
            # =================================================
            if 1:
                self.autoencoder_input_shape = input_autoencoder_shape
                self.autoencoder_output_shape = output_autoencoder_shape
                self.encoder_input_shape = input_encoder_shape
                self.encoder_output_shape = output_encoder_shape
                self.decoder_input_shape = input_decoder_shape
                self.decoder_output_shape = output_decoder_shape
            else:
                self.autoencoder_input_shape = autoencoder.input_shape
                self.autoencoder_output_shape = autoencoder.output_shape
                self.encoder_input_shape = encoder.input_shape
                self.encoder_output_shape = encoder.output_shape
                self.decoder_input_shape = decoder.input_shape
                self.decoder_output_shape = decoder.output_shape




    """
     Compile the model before training
    """
    def compile(self):
        ypixels_model = self.training_image_shape[1]
        xpixels_model = self.training_image_shape[2]
        if ypixels_model * xpixels_model != self.encoder_input_shape[1]:
            raise Exception("Invalid input size! height * width != encoder_input_shape[1], current input shape is {0} * {1} and encoder_input_shape is {2}".format(ypixels_model, xpixels_model, self.encoder_input_shape))
        self.autoencoder.compile(optimizer = self.optimizer, loss = self.loss)
        self.is_compiled = True


    """
     Retrieve input and output dimensions
    """
    def get_encoder_input_shape(self):
        return self.input_enc_shape
    def get_encoder_output_shape(self):
        return self.output_enc_shape
    def get_decoder_input_shape(self):
        return self.input_dec_shape
    def get_decoder_output_shape(self):
        return self.output_dec_shape



# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================



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
    # Train model (or load it)
    # =================================================
    if train_model:
        start_time = time.time()  # start timer

        # Start the report
        run_report = open(report_filename, "w")  # write report
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

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create layers
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create hidden encoding layers
        input_img = Input(shape=(x_train.shape[1],))
        encoded = Dense(encode_dim, activation='relu')(input_img)

        # Create hidden decoding layers
        decoded = Dense(x_train.shape[1], activation='sigmoid')(encoded)

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
