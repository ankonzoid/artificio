"""
 AE.py  (author: Anson Wong / git: ankonzoid)

 Autoencoder class (for both simple & convolutional)
"""
from .AE_base import AEBase

import datetime, os, platform, pylab, time
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model

class AE(AEBase):

    def __init__(self):

        # ======================================
        # Necessary parameters
        # ======================================

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

        # Encoding/decoding functions
        self.encode = None
        self.decode = None
        self.encoder_decode = None

        # Report text
        self.start_time = None
        self.current_time = None

        # ======================================
        # Parameters for architecture
        # ======================================
        self.model_name = None


        # ======================================
        # Parameters for report
        # ======================================



        super().__init__()


    ### ====================================================================
    ###
    ### Training
    ###
    ### ====================================================================

    """
     Train the autoencoder
    """
    def train(self, x_train, x_test, n_epochs=50, batch_size=256):
        self.autoencoder.fit(x_train, x_train,
                             epochs = n_epochs,
                             batch_size = batch_size,
                             shuffle = True,
                             validation_data = (x_test, x_test))
    
    ### ====================================================================
    ###
    ### Encoding / Decoding
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

    """
     Encode then decode 
    """
    def encode_decode(self, img):
        if not self.is_compiled:
            raise Exception("Not compiled yet!")
        x_encoded = self.encoder.predict(img)
        img_decoded = self.decoder.predict(x_encoded)
        return img_decoded

    ### ====================================================================
    ###
    ### Model IO
    ###
    ### ====================================================================

    """
     Load model architecture and weights of autoencoder, encoder, and decoded
    """
    def load_model(self, dictfn):

        print("Loading models...")
        # Set model filenames
        autoencoder_filename = dictfn["autoencoder_filename"]
        encoder_filename = dictfn["encoder_filename"]
        decoder_filename = dictfn["decoder_filename"]

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
     Save model architecture and weights to file
    """
    def save_model(self, dictfn):

        print("Saving models...")
        self.autoencoder.save(dictfn["autoencoder_filename"])
        self.encoder.save(dictfn["encoder_filename"])
        self.decoder.save(dictfn["decoder_filename"])


    ### ====================================================================
    ###
    ### Architecture compilation
    ###
    ### ====================================================================

    """
     Set name
    """
    def configure(self, model_name=None):
        self.model_name = model_name

    """
     Set neural network architecture
    """
    def set_arch(self, input_shape=None, output_shape=None):

        ###
        ### Create layers based on model name
        ###

        if self.model_name == "simpleAE":

            encode_dim = 128  # only single hidden layer

            ### Encoding hidden layers ###
            input_img = Input(shape=input_shape)
            encoded = Dense(encode_dim, activation='relu')(input_img)

            ### Decoding hidden layers ###
            decoded = Dense(output_shape[0], activation='sigmoid')(encoded)

        elif self.model_name == "convAE":

            n_hidden_1 = 16  # 1st hidden layer
            n_hidden_2 = 8  # 2nd hidden layer
            n_hidden_3 = 8  # 3rd hidden layer

            convkernel = (3, 3)  # convolution (uses filters): n_filters_1 -> n_filters_2
            poolkernel = (2, 2)  # pooling (down/up samples image): ypix -> ypix/ypoolkernel, xpix -> xpix/ypoolkernel

            ### Encoding hidden layers ###
            input_img = Input(shape=input_shape)  # input layer (ypixels, xpixels, n_channels)
            x = Conv2D(n_hidden_1, convkernel, activation='relu', padding='same')(input_img)
            x = MaxPooling2D(poolkernel, padding='same')(x)
            x = Conv2D(n_hidden_2, convkernel, activation='relu', padding='same')(x)
            x = MaxPooling2D(poolkernel, padding='same')(x)
            x = Conv2D(n_hidden_3, convkernel, activation='relu', padding='same')(x)
            encoded = MaxPooling2D(poolkernel, padding='same')(x)  # encoding layer

            ### Decoding hidden layers ###
            x = Conv2D(n_hidden_3, convkernel, activation='relu', padding='same')(encoded)
            x = UpSampling2D(poolkernel)(x)
            x = Conv2D(n_hidden_2, convkernel, activation='relu', padding='same')(x)
            x = UpSampling2D(poolkernel)(x)
            x = Conv2D(n_hidden_1, convkernel, activation='relu')(x)
            x = UpSampling2D(poolkernel)(x)
            decoded = Conv2D(output_shape[2], convkernel, activation='sigmoid', padding='same')(x)  # output layer

        else:
            raise Exception("Invalid model name given!")


        ###
        ### Create models
        ###

        ### Create autoencoder model ###
        autoencoder = Model(input_img, decoded)
        print("\n\nautoencoder.summary():")
        print(autoencoder.summary())

        # Set encoder input/output dimensions
        input_autoencoder_shape = autoencoder.layers[0].input_shape[1:]
        output_autoencoder_shape = autoencoder.layers[-1].output_shape[1:]


        ### Create encoder model ###
        encoder = Model(input_img, encoded)  # set encoder
        print("\n\nencoder.summary():")
        print(encoder.summary())

        # Set encoder input/output dimensions
        input_encoder_shape = encoder.layers[0].input_shape[1:]
        output_encoder_shape = encoder.layers[-1].output_shape[1:]


        ### Create decoder model ###
        decoded_input = Input(shape=output_encoder_shape)
        if self.model_name == 'simpleAE':
            decoded_output = autoencoder.layers[-1](decoded_input)  # single layer
        elif self.model_name == 'convAE':
            decoded_output = autoencoder.layers[-7](decoded_input)  # Conv2D
            decoded_output = autoencoder.layers[-6](decoded_output)  # UpSampling2D
            decoded_output = autoencoder.layers[-5](decoded_output)  # Conv2D
            decoded_output = autoencoder.layers[-4](decoded_output)  # UpSampling2D
            decoded_output = autoencoder.layers[-3](decoded_output)  # Conv2D
            decoded_output = autoencoder.layers[-2](decoded_output)  # UpSampling2D
            decoded_output = autoencoder.layers[-1](decoded_output)  # Conv2D
        else:
            raise Exception("Invalid model name given!")
        decoder = Model(decoded_input, decoded_output)
        print("\n\ndecoder.summary():")
        print(decoder.summary())

        # Set encoder input/output dimensions
        input_decoder_shape = decoder.layers[0].input_shape[1:]
        output_decoder_shape = decoder.layers[-1].output_shape[1:]


        ###
        ### Assign models
        ###

        self.autoencoder = autoencoder  # untrained
        self.encoder = encoder  # untrained
        self.decoder = decoder  # untrained

        # Set model input/output dimensions
        # We provide two options for getting these numbers:
        # - via the individual layers
        # - via the models
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
    def compile(self, loss="binary_crossentropy", optimizer="adam"):
        self.autoencoder.compile(optimizer=optimizer, loss=loss)
        self.is_compiled = True

    ### ====================================================================
    ###
    ### Naming conventions
    ###
    ### ====================================================================
    """
     Naming conventions of filenames
    """
    def generate_naming_conventions(self, model_name, output_dir):

        autoencoder_filename = os.path.join(output_dir, model_name + "_autoencoder.h5")
        encoder_filename = os.path.join(output_dir, model_name + "_encoder.h5")
        decoder_filename = os.path.join(output_dir, model_name + "_decoder.h5")
        plot_filename_pdf = os.path.join(output_dir, model_name + "_plot.pdf")
        plot_filename_png = os.path.join(output_dir, model_name + "_plot.png")
        report_filename = os.path.join(output_dir, model_name + "_report.txt")

        dictfn = {
            "autoencoder_filename": autoencoder_filename,
            "encoder_filename": encoder_filename,
            "decoder_filename": decoder_filename,
            "plot_filename_pdf": plot_filename_pdf,
            "plot_filename_png": plot_filename_png,
            "report_filename": report_filename
        }

        return dictfn

    ### ====================================================================
    ###
    ### Saving plots
    ###
    ### ====================================================================
    def plot_save_reconstruction(self, x_data_test, img_shape, dictfn, n_plot=10):

        ypixels = img_shape[0]
        xpixels = img_shape[1]
        n_channels = 3

        # Extract the subset of test data to reconstruct and plot
        n = min(len(x_data_test), n_plot)
        x_data_test_plot = x_data_test[np.arange(n)]

        # Perform reconstructions on test data
        x_data_test_plot_reconstructed = self.encode_decode(x_data_test_plot)

        # Create plot to save
        plt.figure(figsize=(20, 4))
        for i in range(n):

            # Plot original image
            ax = plt.subplot(2, n, i + 1)
            img_show_test = \
                x_data_test_plot[i].reshape((ypixels, xpixels, n_channels))
            plt.imshow(img_show_test)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Plot reconstructed image
            ax = plt.subplot(2, n, n + i + 1)
            img_show_test_reconstructed = \
                x_data_test_plot_reconstructed[i].reshape((ypixels, xpixels, n_channels))
            plt.imshow(img_show_test_reconstructed)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        pylab.savefig(dictfn["plot_filename_pdf"])  # save pdf plot
        pylab.savefig(dictfn["plot_filename_png"])  # save png plot
        if 0:
            plt.show()

    ### ====================================================================
    ###
    ### Run report text log
    ###
    ### ====================================================================

    """
     Clean start the report
    """
    def start_report(self, dictfn):
        report_filename = dictfn["report_filename"]
        self.start_time = time.time()
        self.current_time = self.start_time

        # Start report
        report = open(report_filename, "w")
        report.write(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
        report.write("\n\n")
        report.write("Start time = {0}\n".format(0))
        report.write("\n")
        report.write("  platform: {0}\n".format(platform.machine()))
        report.write("  version: {0}\n".format(platform.version()))
        report.write("  system: {0}\n".format(platform.system()))
        report.write("  processor: {0}\n".format(platform.processor()))
        report.write("\n")
        report.write("  autoencoder_filename: {0}\n".format(dictfn["autoencoder_filename"]))
        report.write("  encoder_filename: {0}\n".format(dictfn["encoder_filename"]))
        report.write("  decoder_filename: {0}\n".format(dictfn["decoder_filename"]))
        report.write("  plot_filename_png: {0}\n".format(dictfn["plot_filename_png"]))
        report.write("  plot_filename_pdf: {0}\n".format(dictfn["plot_filename_pdf"]))
        report.write("  report_filename: {0}".format(dictfn["report_filename"]))
        report.close()

    """
     Append architecture to report
    """
    def append_arch_report(self, dictfn):
        report_filename = dictfn["report_filename"]
        if self.start_time == None or self.current_time == None:
            raise Exception("Start report before appending!")

        report = open(report_filename, "a")  # append report
        report.write("\n\n")
        report.write("  autoencoder_input_shape: {0}\n".format(self.autoencoder_input_shape))
        report.write("  autoencoder_output_shape: {0}\n".format(self.autoencoder_output_shape))
        for i in range(len(self.autoencoder.layers)):
            report.write("    autoencoder.layers[{0}]: input={1}, output={2}\n".format(i,
                self.autoencoder.layers[i].input_shape[1:],
                self.autoencoder.layers[i].output_shape[1:]))
        report.write("\n")

        report.write("  encoder_input_shape: {0}\n".format(self.encoder_input_shape))
        report.write("  encoder_output_shape: {0}\n".format(self.encoder_output_shape))
        for i in range(len(self.encoder.layers)):
            report.write("    encoder.layers[{0}]: input={1}, output={2}\n".format(i,
                self.encoder.layers[i].input_shape[1:],
                self.encoder.layers[i].output_shape[1:]))
        report.write("\n")

        report.write("  decoder_input_shape: {0}\n".format(self.decoder_input_shape))
        report.write("  decoder_output_shape: {0}\n".format(self.decoder_output_shape))
        for i in range(len(self.decoder.layers)):
            report.write("    decoder.layers[{0}]: input={1}, output={2}\n".format(i,
                self.decoder.layers[i].input_shape[1:],
                self.decoder.layers[i].output_shape[1:]))

        report.close()


    """
     Append custom message to report
    """
    def append_message_report(self, dictfn, custom_message):
        report_filename = dictfn["report_filename"]
        if self.start_time == None or self.current_time == None:
            raise Exception("Start report before appending!")
        self.current_time = time.time()  # record new current time

        report = open(report_filename, "a")  # append report
        report.write("\n{0} = {1}\n\n".format(custom_message, self.current_time - self.start_time))
        report.close()



def main():
    pass

#
# Driver file
#
if __name__ == '__main__':
    main()
