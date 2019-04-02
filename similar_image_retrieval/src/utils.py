"""

 utils.py (author: Anson Wong / git: ankonzoid)

 Image utilities class that helps with image data IO and processing.

"""
import os, glob, random, errno
import numpy as np
import scipy.misc
from multiprocessing import Pool
import matplotlib.pyplot as plt

"""
 Finds the first unique k elements (based on lowest distance) of lists 'indices' and 'distances'
"""
def find_topk_unique(indices, distances, k):

    # Sort by ascending distance
    i_sort_1 = np.argsort(distances)
    distances_sorted = distances[i_sort_1]
    indices_sorted = indices[i_sort_1]

    window = np.array(indices_sorted[:k], dtype=int)  # collect first k elements for window intialization
    window_unique, j_window_unique = np.unique(window, return_index=True)  # find unique window values and indices
    j = k  # track add index when there are not enough unique values in the window
    # Run while loop until window_unique has k elements
    while len(window_unique) != k:
        # Append new index and value to the window
        j_window_unique = np.append(j_window_unique, [j])  # append new index
        window = np.append(window_unique, [indices_sorted[j]])  # append new value
        # Update the new unique window
        window_unique, j_window_unique_temp = np.unique(window, return_index=True)
        j_window_unique = j_window_unique[j_window_unique_temp]
        # Update add index
        j += 1

    # Sort the j_window_unique (not sorted) by distances and get corresponding
    # top-k unique indices and distances (based on smallest distances)
    distances_sorted_window = distances_sorted[j_window_unique]
    indices_sorted_window = indices_sorted[j_window_unique]
    u_sort = np.argsort(distances_sorted_window)  # sort

    distances_top_k_unique = distances_sorted_window[u_sort].reshape((1, -1))
    indices_top_k_unique = indices_sorted_window[u_sort].reshape((1, -1))

    return indices_top_k_unique, distances_top_k_unique

"""
 Checks if a list has unique elements
"""
def is_unique(vec):
    n_vec = len(vec)
    n_vec_unique = len(np.unique(vec))
    return (n_vec == n_vec_unique)



class PlotUtils(object):

    """
     Plots images in 2 rows: top row is query, bottom row is answer
    """
    def plot_query_answer(self, x_query=None, x_answer=None, filename=None, gray_scale=False, n=5):

        # n = maximum number of answer images to provide
        plt.figure(figsize=(2*n, 4))

        # Plot query images
        for j, img in enumerate(x_query):
            if(j >= n):
                break
            ax = plt.subplot(2, n, j + 1)  # display original
            plt.imshow(img)
            if gray_scale:
                plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(4)  # increase border thickness
                ax.spines[axis].set_color('black')  # set to black
            ax.set_title("query",  fontsize=14)  # set subplot title

        # Plot answer images
        for j, img in enumerate(x_answer):
            if (j >= n):
                break

            ax = plt.subplot(2, n, n + j + 1)  # display original
            plt.imshow(img)
            if gray_scale:
                plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1)  # set border thickness
                ax.spines[axis].set_color('black')  # set to black
            ax.set_title("rec %d" % (j+1), fontsize=14)  # set subplot title

        if filename == None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight')
        plt.close()

class ImageUtils(object):

    def __init__(self):
        # Run settings
        self.img_shape = None
        self.flatten_before_encode = False

        # Directories
        self.query_dir = None
        self.answer_dir = None

        self.img_train_raw_dir = None
        self.img_inventory_raw_dir = None
        self.img_train_dir = None
        self.img_inventory_dir = None


    ############################################################
    ###
    ###
    ### External functions
    ###
    ###
    ############################################################

    """
     Set configuration
    """
    def configure(self, info):
        # Run settings
        self.img_shape = info["img_shape"]
        self.flatten_before_encode = info["flatten_before_encode"]

        # Directories
        self.query_dir = info["query_dir"]
        self.answer_dir = info["answer_dir"]

        self.img_train_raw_dir = info["img_train_raw_dir"]
        self.img_inventory_raw_dir = info["img_inventory_raw_dir"]
        self.img_train_dir = info["img_train_dir"]
        self.img_inventory_dir = info["img_inventory_dir"]


    """
     Load raw images from a given directory, resize them, and save them
    """
    def raw2resized_load_save(self, raw_dir=None, processed_dir=None, img_shape=None):

        (ypixels_force, xpixels_force) = img_shape
        gray_scale = False

        # Extract filenames from dir
        raw_filenames_list, n_files = self.extract_filenames(raw_dir, 1)

        for i, raw_filename in enumerate(raw_filenames_list):

            # Read raw image
            img_raw = self.read_img(raw_filename, gray_scale=gray_scale)

            # Process image
            img_resized = self.force_resize_img(img_raw, ypixels_force, xpixels_force)

            # Save processed image
            name, tag = self.extract_name_tag(raw_filename)
            processed_shortname = name + "_resized." + tag
            processed_filename = os.path.join(processed_dir, processed_shortname)
            self.save_img(img_resized, processed_filename)

            # Print process progress
            print("[{0}/{1}] Resized and saved to '{2}'...".format(
                i+1, n_files, processed_filename))

    """
     Read a raw directory of images, and load as numpy array to memory
    """
    def raw2resizednorm_load(self, raw_dir=None, img_shape=None):

        (ypixels_force, xpixels_force) = img_shape
        gray_scale = False

        # Extract filenames from dir
        raw_filenames_list, n_files = self.extract_filenames(raw_dir, 1)

        img_list = []
        for i, raw_filename in enumerate(raw_filenames_list):

            # Read raw image
            img_raw = self.read_img(raw_filename, gray_scale=gray_scale)

            # Resize image (if not of shape (ypixels_force, xpixels_force))
            img_resized = img_raw
            if img_raw.shape[:2] != img_shape:
                img_resized = self.force_resize_img(img_resized, ypixels_force, xpixels_force)

            # Normalize image
            img_resizednorm = self.normalize_img_data(img_resized)

            # Append processed image to image list
            img_list.append(img_resizednorm)

            # Print process progress
            print("[{0}/{1}] Loaded and processed '{2}'...".format(
                i + 1, n_files, raw_filename))

        # Convert image list to numpy array
        img_list = np.array(img_list)

        # Make tests
        if img_list.shape[0] != n_files:
            raise Exception("Inconsistent number of loading images!")
        if img_list.shape[1] != ypixels_force:
            raise Exception("Inconsistent ypixels loading images!")
        if img_list.shape[2] != xpixels_force:
            raise Exception("Inconsistent xpixels loading images!")
        if img_list.shape[3] != 3:
            raise Exception("Inconsistent RGB loading images!")

        return img_list, raw_filenames_list

    """
     Split image data set to training and test set using a seeded psuedo 
     random number generator
    """
    def split_train_test(self, x_data, ratio_train_test, seed):

        # Find number of examples we have
        n = len(x_data)

        # Generate a list of random [0,1] numbers
        random.seed(a = seed)
        r_list = [random.random() for x in range(n)]

        # Perform splitting given our splitting ratio
        x_data_train = []; x_data_test = []
        index_train = []; index_test = []
        for i in range(n):

            # Assign to training if: r_list[i] < ratio_train_test
            # Assign to test data if: r_list[i] >= ratio_train_test
            if r_list[i] < ratio_train_test:
                x_data_train.append(x_data[i])
                index_train.append(i)
            else:
                x_data_test.append(x_data[i])
                index_test.append(i)

        # Convert to numpy arrays
        x_data_train = np.array(x_data_train, dtype=float)
        x_data_test = np.array(x_data_test, dtype=float)
        index_train = np.array(index_train, dtype=int)
        index_test = np.array(index_test, dtype=int)

        # Make checks that the sets are not empty
        if len(x_data_train) == 0:
            raise Exception("Split train set is empty!")
        if len(x_data_test) == 0:
            raise Exception("Split test set is empty!")

        return x_data_train, x_data_test, index_train, index_test

    ############################################################
    ###
    ###
    ### Internal functions
    ###
    ###
    ############################################################


    ### =============================================
    ### Load image IO
    ### =============================================

    """
     Read image (grayscale/RGB)
    """
    def read_img(self, img_filename, gray_scale=False):
        if gray_scale:
            img = np.array(scipy.misc.imread(img_filename, flatten=gray_scale))
        else:
            img = np.array(scipy.misc.imread(img_filename, mode='RGB'))
        return img

    """
     This function is to be paired with Pool
    """
    def _read_img_parallel(self, zipped_enumerated_filenames_list):
        filenames_list, gray_scale = zipped_enumerated_filenames_list
        print("Reading {0} (gray_scale={1})".format(filenames_list, gray_scale))
        img = self.read_img(filenames_list, gray_scale=gray_scale)  # greyscale image (img.shape[0]=y, img.shape[1]=x)
        return img

    """
     Load images from directory (parallelized to cores)
    """
    def load_images_parallel(self, dir, ratio_training_test, seed, n_cores):
        frac_training_use = 1
        gray_scale = False
        index_train = []; index_test = []
        x_train = []; x_test = []
        filenames_list, n_files = self.extract_filenames(dir, frac_training_use)

        # Set up seed, and create a list of pre-determined random numbers (between [0,1]) following this seed
        random.seed(a = seed)
        r_list = [random.random() for x in range(n_files)]  # list of pseudo random [0,1]'s

        # Read images in parallel
        gray_scale_vec = [gray_scale] * n_files
        pool = Pool(n_cores)
        img_list = pool.map(self._read_img_parallel, zip(filenames_list, gray_scale_vec))  # pool.map(f, input_f)
        pool.close()  # close the pool
        pool.join()  # join the pool
        img_mat = np.array(img_list)
        ypixels_data = img_mat.shape[1]
        xpixels_data = img_mat.shape[2]

        # Append each sample to the training or test set
        for i in range(len(img_mat)):
            if r_list[i] < ratio_training_test:
                x_train.append(img_mat[i])
                index_train.append(i)
            else:
                x_test.append(img_mat[i])
                index_test.append(i)

        # Depending on gray_scale, we reshape to 3 or 4 dimensional array
        if gray_scale:
            x_train = np.array(x_train).reshape((-1, ypixels_data, xpixels_data))  # convert to 3-dim numpy
            x_test = np.array(x_test).reshape((-1, ypixels_data, xpixels_data))  # convert to 3-dim numpy
        else:
            x_train = np.array(x_train).reshape((-1, ypixels_data, xpixels_data, 3))  # convert to 4-dim numpy
            x_test = np.array(x_test).reshape((-1, ypixels_data, xpixels_data, 3))  # convert to 4-dim numpy

        index_train = np.array(index_train, dtype=int)
        index_test = np.array(index_test, dtype=int)
        return x_train, x_test, index_train, index_test, filenames_list


    ### =============================================
    ### Save image IO
    ### =============================================

    """
     Save image to a directory
      - If img.shape = (ypixels, xpixels), outputs greyscaled image
      - If img.shape = (ypixels, xpixels, 3), outputs RGB image
    """
    def save_img(self, img, save_filename):
        scipy.misc.imsave(save_filename, img)
        return

    ### =============================================
    ### Image processing
    ### =============================================

    """
     Flatten image data and flatten pixel dimensions
    """
    def flatten_img_data(self, x_data):
        n_data = x_data.shape[0]
        flatten_dim = np.prod(x_data.shape[1:])
        x_data_flatten = x_data.reshape((n_data, flatten_dim))
        return x_data_flatten

    """
     Normalize image data (no reshaping)
    """
    def normalize_img_data(self, x_data):
        x_data_norm = x_data.astype('float32') / 255.  # normalize values [0,1]
        return x_data_norm

    """
     Force resize image data to given (ypixels, xpixels)
    """
    def force_resize_img(self, img, ypixels_force, xpixels_force):
        img_resized = np.array(scipy.misc.imresize(img, (ypixels_force, xpixels_force)))
        return img_resized


    ### =============================================
    ### Filename IO
    ### =============================================

    """
     Extract name and tag from a filename
    """
    def extract_name_tag(self, full_filename):
        shortname = full_filename[full_filename.rfind("/") + 1:]  # filename (within its last directory)
        shortname_split = shortname.split('.')
        name = ''
        n_splits = len(shortname_split)
        for i, x in enumerate(shortname_split):
            if i == 0:
                name = name + x
            elif i == n_splits - 1:
                break
            else:
                name = name + '.' + x
        tag = shortname_split[-1]
        return name, tag

    """
     Extract all filenames inside a given directory (with fraction option):
      - list of filenames
    """
    def extract_filenames(self, dir, frac_take):
        filenames_list = glob.glob(dir + "/*")
        n_files = len(filenames_list)
        if n_files == 0:
            raise Exception("There are no files in {0}!".format(dir))
        if frac_take != 1:
            n_files = int(frac_take * n_files)
            filenames_list = filenames_list[:n_files]
        return filenames_list, n_files

    ### =============================================
    ### Directory IO
    ### =============================================

    """
     Force make directory
    """
    def make_dir(self, dir):
        try:
            os.makedirs(dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise Exception("Unexpected error in making directory!")

