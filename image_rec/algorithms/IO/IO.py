import numpy as np
import random
import glob
import scipy.misc
import os
import errno
import matplotlib.pyplot as plt
from multiprocessing import Pool

# normalize image and reshapes (for single layer AE) to:
# - (n, ypixels, xpixels)
def normalize_flatten_img_data(x_data):
    n_data = x_data.shape[0]
    n_flatten_dim = np.prod(x_data.shape[1:])
    x_data = x_data.astype('float32') / 255.  # normalize values [0,255] -> [0,1]
    x_data = x_data.reshape((n_data, n_flatten_dim))  # flatten (ypixels, xpixels) -> (ypixels*xpixels)
    return x_data

# normalize image and reshapes (for Conv AE) to:
# - (n, ypixels, xpixels, n_channels)
def normalize_reshape_img_data(x_data, n_channels):
    n_data = x_data.shape[0]
    ypixels = x_data.shape[1]
    xpixels = x_data.shape[2]
    x_norm = x_data.astype('float32') / 255.  # normalize values [0,255] -> [0,1]
    x_data_norm_reshaped = np.reshape(x_norm, (n_data, ypixels, xpixels, n_channels))  # flatten (ypixels, xpixels) -> (ypixels*xpixels)
    return x_data_norm_reshaped

# loads images from a directory in:
# - (n, ypixels, xpixels) if gray_scale == True
# - (n, ypixels, xpixels, n_channels) if gray_scale == False
def load_all_images_from_dir(dir, gray_scale):
    index_list = []
    x_list = []
    ypixels_data = None
    xpixels_data = None
    filenames_list, n_files = extract_filenames_in_dir(dir, 1)
    if n_files == 0:
        raise Exception("There are no files in {0}!".format(dir))

    for i, full_filename_i in enumerate(filenames_list):
        # Read single image:
        # - if gray_scale, we get (ypixels, xpixels)
        # - if rgb, we get (ypixels, xpixels, n_channels)
        img = read_img(full_filename_i, gray_scale=gray_scale)  # greyscale image (img.shape[0]=y, img.shape[1]=x)

        print("[{0}/{1}] Reading image {2}...".format(i+1,n_files,full_filename_i))
        # Take ypixels_data and xpixels_data from 1st image
        if i==0:
            ypixels_data = img.shape[0]
            xpixels_data = img.shape[1]
        else:
            if img.shape[0]!=ypixels_data or img.shape[1]!=xpixels_data:
                raise Exception("Inconsistent image sizes in {0}!".format(dir))

        x_list.append(img)
        index_list.append(i)

    # Depending on gray_scale, we reshape to 3 or 4 dimensional array
    if gray_scale:
        x_list = np.array(x_list).reshape((-1, ypixels_data, xpixels_data))  # convert to 3-dim numpy
    else:
        n_channels = 3
        x_list = np.array(x_list).reshape((-1, ypixels_data, xpixels_data, n_channels))  # convert to 4-dim numpy
    index_list = np.array(index_list, dtype=int)
    return x_list, index_list, filenames_list

# loads images from a directory in:
# - (n, ypixels, xpixels) if gray_scale == True
# - (n, ypixels, xpixels, n_channels) if gray_scale == False
def load_images_from_dir(dir, ratio_training_test, frac_training_use, n_clone, seed, gray_scale):
    index_train = []
    index_test = []
    x_train = []
    x_test = []
    ypixels_data = None
    xpixels_data = None
    filenames_list, n_files = extract_filenames_in_dir(dir, frac_training_use)
    if n_files == 0:
        raise Exception("There are no files in {0}!".format(dir))

    # Set up seed, and create a list of pre-determined random numbers (between [0,1]) following this seed
    random.seed(a = seed)
    r_list = [random.random() for x in range(n_files)]  # list of pseudo random [0,1]'s

    for i, full_filename_i in enumerate(filenames_list):
        # Read single image:
        # - if gray_scale, we get (ypixels, xpixels)
        # - if rgb, we get (ypixels, xpixels, n_channels)
        img = read_img(full_filename_i, gray_scale=gray_scale)  # greyscale image (img.shape[0]=y, img.shape[1]=x)

        print("[{0}/{1}] Reading image {2}...".format(i+1,n_files,full_filename_i))
        # Take ypixels_data and xpixels_data from 1st image
        if i==0:
            ypixels_data = img.shape[0]
            xpixels_data = img.shape[1]
        else:
            if img.shape[0]!=ypixels_data or img.shape[1]!=xpixels_data:
                raise Exception("Inconsistent image sizes in {0}!".format(dir))

        # Assign to training or test set
        if r_list[i] < ratio_training_test:
            for j in range(n_clone):
                x_train.append(img)
                index_train.append(i)
        else:
            for j in range(n_clone):
                x_test.append(img)
                index_test.append(i)

    # Depending on gray_scale, we reshape to 3 or 4 dimensional array
    if gray_scale:
        x_train = np.array(x_train).reshape((-1, ypixels_data, xpixels_data))  # convert to 3-dim numpy
        x_test = np.array(x_test).reshape((-1, ypixels_data, xpixels_data))  # convert to 3-dim numpy
    else:
        n_channels = 3
        x_train = np.array(x_train).reshape((-1, ypixels_data, xpixels_data, n_channels))  # convert to 4-dim numpy
        x_test = np.array(x_test).reshape((-1, ypixels_data, xpixels_data, n_channels))  # convert to 4-dim numpy

    index_train = np.array(index_train, dtype=int)
    index_test = np.array(index_test, dtype=int)
    return x_train, x_test, index_train, index_test, filenames_list

# this function is to be paired with Pool
def read_img_parallel(zipped_enumerated_filenames_list):
    filenames_list, gray_scale = zipped_enumerated_filenames_list
    print("Reading {0} (gray_scale={1})".format(filenames_list, gray_scale))
    img = read_img(filenames_list, gray_scale=gray_scale)  # greyscale image (img.shape[0]=y, img.shape[1]=x)
    return img

# loads images from a directory (parallel) in:
# - (n, ypixels, xpixels) if gray_scale == True
# - (n, ypixels, xpixels, n_channels) if gray_scale == False
def load_images_from_dir_parallel(dir, ratio_training_test, frac_training_use, n_clone, seed, gray_scale, n_cores):
    index_train = []
    index_test = []
    x_train = []
    x_test = []
    filenames_list, n_files = extract_filenames_in_dir(dir, frac_training_use)
    if n_files == 0:
        raise Exception("There are no files in {0}!".format(dir))

    # Set up seed, and create a list of pre-determined random numbers (between [0,1]) following this seed
    random.seed(a=seed)
    r_list = [random.random() for x in range(n_files)]  # list of pseudo random [0,1]'s

    # Read images in parallel
    gray_scale_vec = [gray_scale] * n_files
    pool = Pool(n_cores)
    img_list = pool.map(read_img_parallel, zip(filenames_list, gray_scale_vec))  # pool.map(f, input_f)
    pool.close()  # close the pool
    pool.join()  # join the pool
    img_mat = np.array(img_list)
    ypixels_data = img_mat.shape[1]
    xpixels_data = img_mat.shape[2]

    # Append each sample to the training or test set
    for i in range(len(img_mat)):
        if r_list[i] < ratio_training_test:
            for j in range(n_clone):
                x_train.append(img_mat[i])
                index_train.append(i)
        else:
            for j in range(n_clone):
                x_test.append(img_mat[i])
                index_test.append(i)

    # Depending on gray_scale, we reshape to 3 or 4 dimensional array
    if gray_scale:
        x_train = np.array(x_train).reshape((-1, ypixels_data, xpixels_data))  # convert to 3-dim numpy
        x_test = np.array(x_test).reshape((-1, ypixels_data, xpixels_data))  # convert to 3-dim numpy
    else:
        n_channels = 3
        x_train = np.array(x_train).reshape((-1, ypixels_data, xpixels_data, n_channels))  # convert to 4-dim numpy
        x_test = np.array(x_test).reshape((-1, ypixels_data, xpixels_data, n_channels))  # convert to 4-dim numpy

    index_train = np.array(index_train, dtype=int)
    index_test = np.array(index_test, dtype=int)
    return x_train, x_test, index_train, index_test, filenames_list


# Extract all filenames inside a particular directory
# option of extracting only the first frac_take filenames
def extract_filenames_in_dir(dir, frac_take):
    filenames_list = glob.glob(dir + "/*")
    n_files = len(filenames_list)
    print(n_files)
    if frac_take != 1:
        n_files = int(frac_take*n_files)
        filenames_list = filenames_list[:n_files]
    return  filenames_list, n_files

def extract_name_tag_filename(full_filename):
    shortname = full_filename[full_filename.rfind("/") + 1:]  # filename (within its last directory)
    shortname_split = shortname.split('.')
    name = ''
    n_splits = len(shortname_split)
    for i, x in enumerate(shortname_split):
        if i==0:
            name = name + x
        elif i==n_splits-1:
            break
        else:
            name = name + '.' + x
    tag = shortname_split[-1]
    return name, tag

def read_img(img_filename, gray_scale=False):
    if gray_scale:
        img = np.array(scipy.misc.imread(img_filename, flatten=gray_scale))
    else:
        img = np.array(scipy.misc.imread(img_filename, mode='RGB'))
    return img

def force_resize_img(img, height, width):
    img_resized = np.array(scipy.misc.imresize(img, (height, width)))
    return img_resized

def save_processed_img(img_processed, dir, name, tag, addon_tag):
    full_filename_processed = dir + '/' + name + addon_tag + '.' + tag
    scipy.misc.imsave(full_filename_processed, img_processed)
    return

def make_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise