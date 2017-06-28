"""
 image_transformer.py
"""
from .transform_base import TransformerBase
import numpy as np
import scipy.misc
import os
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class ImageTransformer(TransformerBase):
    def __init__(self):
        self.output_shape = None
        self.encoder = None
        super().__init__()

    def register_encoder(self, encoder=None):
        self.encoder = encoder

    def configure(self, output_shape):
        self.output_shape = output_shape

    def transform_one(self, file_path, grey_scale=False, flatten=True):
        if self.output_shape is None:
            raise Exception("output shape is not configured")
        ret = self._read_data_parallel([file_path], gray_scale=grey_scale, flatten=flatten)
        if self.encoder is not None:
            return self.encoder.predict(ret)
        return ret

    def transform_all(self, folder_name, grey_scale=False, batch_size=256, multi_thread=True, flatten=False):
        if self.output_shape is None:
            raise Exception("output shape is not configured")
        if multi_thread:
            onlyfiles = [os.path.join(folder_name, f) for f in listdir(folder_name) if isfile(join(folder_name, f))]
            for chunk in chunks(onlyfiles, batch_size):
                # Read the data without flattening it, as we want to feed it into encoder
                data = self._read_data_parallel(chunk, gray_scale=grey_scale, batch_size=batch_size, flatten=flatten)
                yield data  # do not apply encoder to this, output raw data

    def _read_data_parallel(self, onlyfiles, gray_scale=False, batch_size=256, flatten=True):
        patch = zip(onlyfiles, [gray_scale]*len(onlyfiles), [self.output_shape]*len(onlyfiles))
        p = Pool(8)
        batch = p.map(ImageTransformer._read_file_worker, patch)
        p.close()
        p.join()
        # nothing being removed ...
        batch = [x for x in batch if x is not None]
        batch = np.array(batch)
        if flatten:
            reshape_size = (batch.shape[0], np.prod(batch.shape[1:]))
            batch = batch.reshape(reshape_size)
        batch = batch.astype('float32') / 255.
        return batch

    @staticmethod
    def _read_file_worker(patch):
        file_path, gray_scale, output_shape = patch
        if file_path[-3:] not in ['jpg', 'png', 'jpeg']:
            return
        img = ImageTransformer._read_img(file_path, gray_scale=gray_scale)
        img = np.array(scipy.misc.imresize(img, output_shape))
        return img

    # read_img: read image, and convert to numpy
    @staticmethod
    def _read_img(img_filename, gray_scale=False):
        if gray_scale is True:
            img = np.array(scipy.misc.imread(img_filename, flatten=gray_scale))
        else:
            img = np.array(scipy.misc.imread(img_filename, mode='RGB'))
        return img
