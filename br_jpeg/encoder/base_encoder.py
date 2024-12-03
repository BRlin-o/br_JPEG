from abc import ABC, abstractmethod

class EncoderBase(ABC):
    def __init__(self, image_path=None, DEBUG=False):
        self.DEBUG = DEBUG
        self.init()

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def encode(self, image_path, quality, save_path, to_gray):
        pass
