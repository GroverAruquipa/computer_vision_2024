from abc import ABC, abstractmethod


class WaitStrategy(ABC):
    @abstractmethod
    def wait(self):
        raise NotImplementedError()
