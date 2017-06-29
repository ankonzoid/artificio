"""
 Autoencoder base (author: Anson Wong / git: ankonzoid)
"""
from abc import ABC, abstractmethod

class EmbeddingBase(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_training_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_test_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def arch(self, *args, **kwargs):
        pass

    @abstractmethod
    def compile(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_encoder_input_shape(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_encoder_output_shape(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_decoder_input_shape(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_decoder_output_shape(self, *args, **kwargs):
        pass
