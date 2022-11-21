from numpy.typing import NDArray
from typing import Tuple
from abc import ABC, abstractmethod


class DataGenerator(ABC):

    @abstractmethod
    def __init__(self, *args) -> None:
        pass

    @abstractmethod
    def generate_data(self, *args) -> Tuple[NDArray, NDArray]:
        pass
