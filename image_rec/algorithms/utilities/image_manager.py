'''
 image_manager.py
'''
from .image_transformer import ImageTransformer
import numpy as np
import os
from os import listdir
from os.path import isfile, join

class ImageManager(object):
    
    def __init__(self):
        self.db_name = None
        self.flatten = False
        self.output_shape = None
        self.query_folder = None
        self.answer_folder = None
        self.train_db_paths = None
        self.train_bin_db_paths = None
        self.encoder_filename = None

        self.encoder = None  # register_encoder
        self.counter = 0  # load_train_data
        self.vectors = None  # load_train_data
        self.f_idx = {}  # set by build_mapping
        self.idx_f = {}  # set by build_mapping

    def register_encoder(self, encoder=None):
        self.encoder = encoder

    def get_file_name(self, idx):
        return self.idx_f[idx]

    def get_index(self, f_name):
        return self.f_idx[f_name]

    def build_mapping(self):
        if self.train_db_paths is None:
            raise Exception('Config before use')
        mat_file_list = []
        for each_path in self.train_db_paths:
            #print(each_path if 'py' not in each_path else '')
            check = [
                1 for f in listdir(each_path)
                if (f[-3:] == 'jpg' and isfile(join(each_path, f)))
            ]
            mat_file_list += [os.path.join(each_path, f) for f in listdir(each_path) if f[-3:]=='jpg' and isfile(join(each_path, f))]
        self.f_idx = dict(zip(mat_file_list, range(len(mat_file_list))))
        self.idx_f = dict(zip(range(len(mat_file_list)), mat_file_list))


    def configure(self, config):
        self.db_name = config['db_name']
        self.grey_scale = config['grey_scale']
        self.flatten = config['flatten']
        self.output_shape = config['output_shape']
        self.query_folder = config['query_folder']
        self.answer_folder = config['answer_folder']
        self.train_db_paths = config['train_db_paths']
        self.train_bin_db_paths = config['train_bin_db_paths']
        self.encoder_filename = config['encoder_filename']

    def load_raw_data(self, batch_size=5000):
        self.counter = 0
        if isinstance(self.train_db_paths, list):
            for i, folder_i in enumerate(self.train_db_paths):
                self._img_to_numpy_array(folder_path=folder_i, output_shape=self.output_shape,
                                         save_folder_path=self.train_bin_db_paths[i], db_name=self.db_name,
                                         batch_size=batch_size, multi_thread=True,
                                         grey_scale=self.grey_scale, flatten=self.flatten)
        else:
            raise Exception('Unsupported format, only supports str and list (list of paths)')
        return self.vectors

    def _img_to_numpy_array(self, folder_path=None, output_shape=None, save_folder_path=None, db_name=None,
                            batch_size=5000, multi_thread=True, grey_scale = False, flatten=True):
        t = ImageTransformer()
        t.configure(output_shape)
        for chunk in t.transform_all(folder_path, grey_scale=grey_scale, batch_size=batch_size,
                                     multi_thread=multi_thread, flatten=flatten):
            print("Chunk size = {0}".format(chunk.shape))
            ret = chunk  # do not apply encoder on this, take raw chunk
            if self.vectors is None:
                self.vectors = ret
            else:
                self.vectors = np.concatenate((self.vectors, ret))

            chunk_filename = os.path.join(save_folder_path, db_name + "_" + str(self.counter))
            np.save(chunk_filename, ret)
            self.counter += 1

    def load_dataset(self):
        mat_file_list = [os.path.join(self.train_bin_db_paths, f) for f in listdir(self.train_bin_db_paths) if isfile(join(self.train_bin_db_paths, f))]
        all = None
        for mat_file in mat_file_list:
            ds = np.load(mat_file)
            if all is None:
                all = ds
            else:
                all = np.concatenate((all, ds))
        return all

    def encode_raw(self, x_train_raw):
        if self.flatten:
            x_train_raw_flatten = x_train_raw.reshape((x_train_raw.shape[0], np.prod(x_train_raw.shape[1:])))
            x_train_enc_flatten = self.encoder.predict(x_train_raw_flatten)  # -> (n_train, enc_dim)
        else:
            x_train_enc = self.encoder.predict(x_train_raw)  # -> (n_train, enc_dim_y, enc_dim_x, n_channels_enc)
            x_train_enc_flatten = x_train_enc.reshape(
                (x_train_enc.shape[0], np.prod(x_train_enc.shape[1:])))  # -> (n_train, enc_dim)
        return x_train_enc_flatten

    def extract_name_tag_filename(self, full_filename):
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
