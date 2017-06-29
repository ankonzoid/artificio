import os

# naming conventions of folders and filenames
def make_folder_filename_conventions(name_algo, output_dir):

    autoencoder_filename = os.path.join(output_dir, name_algo + "_autoencoder.h5")
    encoder_filename = os.path.join(output_dir, name_algo + "_encoder.h5")
    decoder_filename = os.path.join(output_dir, name_algo + "_decoder.h5")
    plot_filename_pdf = os.path.join(output_dir, name_algo + "_plot.pdf")
    plot_filename_png = os.path.join(output_dir, name_algo + "_plot.png")
    report_filename = os.path.join(output_dir, name_algo + "_report.txt")

    dict = \
    {
        "autoencoder_filename": autoencoder_filename,
        "encoder_filename": encoder_filename,
        "decoder_filename": decoder_filename,
        "plot_filename_pdf": plot_filename_pdf,
        "plot_filename_png": plot_filename_png,
        "report_filename": report_filename
    }

    return dict