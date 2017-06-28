
# naming conventions of folders and filenames
def make_folder_filename_conventions(name_algo, training_heaven_dir, name_data, name_trainingset):

    training_dir = training_heaven_dir + "/" + name_data + "/" + name_trainingset

    model_dir = training_heaven_dir + "/" + name_data + "/" + "MODELS"
    model_subdir = model_dir + "/" + name_algo + "_" + name_data + "_" + name_trainingset
    file_template = model_subdir + "/" + name_algo + "_" + name_data + "_" + name_trainingset

    autoencoder_filename = file_template + "_autoencoder.h5"
    encoder_filename = file_template + "_encoder.h5"
    decoder_filename = file_template + "_decoder.h5"
    plot_filename_pdf = file_template + "_plot.pdf"
    plot_filename_png = file_template + "_plot.png"
    train_report_filename = file_template + "_report.txt"

    dict = \
    {
        "training_dir": training_dir,
        "model_dir": model_dir,
        "model_subdir": model_subdir,
        "autoencoder_filename": autoencoder_filename,
        "autoencoder_checkpoint_filename": autoencoder_checkpoint_filename,
        "encoder_filename": encoder_filename,
        "decoder_filename": decoder_filename,
        "plot_filename_pdf": plot_filename_pdf,
        "plot_filename_png": plot_filename_png,
        "train_report_filename": run_report_filename
    }

    return dict