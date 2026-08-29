from typing import Generic, TypeVar
from abc import ABC, abstractmethod

TRAIN_TYPE = TypeVar("TRAIN_TYPE")
FEAT_TYPE = TypeVar("FEAT_TYPE")
OUT_TYPE = TypeVar("OUT_TYPE")


class BaseModel(ABC, Generic[TRAIN_TYPE, FEAT_TYPE, OUT_TYPE]):
    @abstractmethod
    def train(self, data: TRAIN_TYPE) -> None:
        pass

    @abstractmethod
    def predict(self, data: FEAT_TYPE) -> OUT_TYPE:
        pass

    @abstractmethod
    def evaluate(self, test_data: TRAIN_TYPE) -> dict[str, float]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass
