import logging
import time
from math import floor

from rich.progress import track

from src.domain.wait.wait import WaitStrategy


class TimerWaitStrategy(WaitStrategy):
    def __init__(self, duration: float):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.duration = duration
        self.sleep_time = 0.01


    def wait(self):
        time.time()
        self.logger.info(f"Waiting {self.duration} seconds")
        for _ in track(range(floor(self.duration / self.sleep_time)), description="Waiting..."):
            time.sleep(self.sleep_time)

        self.logger.info("Waiting stopped")
