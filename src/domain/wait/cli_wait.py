import logging

from rich.prompt import Confirm

from src.domain.wait.wait import WaitStrategy


class CliWaitStrategy(WaitStrategy):
    def __init__(self, prompt: str = "Do you whant to proceed? (y|Y)"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.prompt = prompt
        self._proceeded = False

    def wait(self):
        if not Confirm.ask(self.prompt):
            self.logger.error("Execution stoped")
            raise RuntimeError("User stoped execution.")
        self.logger.info("Execution continue")
