"""
 Autoencoder base (author: Anson Wong / git: ankonzoid)
"""
from abc import ABC, abstractmethod

class AEBase(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_arch(self, *args, **kwargs):
        pass

    @abstractmethod
    def compile(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass

    @abstractmethod
    def encode_decode(self, *args, **kwargs):
        pass
