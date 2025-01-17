
import abc
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class AbsAgent(abc.ABC):
    @abc.abstractmethod
    def get_action(self, state: np.ndarray) -> int: ...

    @abc.abstractmethod
    def train(self, *arge, **kwargs) -> Any: ...

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs) -> Any: ...

    @abc.abstractmethod
    def done(self, *args, **kwargs) -> Any: ...

class HyperParams(BaseModel): 
    ...