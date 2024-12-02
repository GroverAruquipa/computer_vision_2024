import logging

from rich.prompt import Confirm
from typing_extensions import override

from src.domain.context import CliWaitConfig
from src.domain.wait.wait import WaitStrategy


class CliWaitStrategy(WaitStrategy):
    PROMPT: str = "Do you whant to proceed?"

    def __init__(self, config: CliWaitConfig):
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self._proceeded: bool = False

    @override
    def wait(self):
        if not Confirm.ask(self.PROMPT, choices=["y", "Y", "n", "N"]):
            self.logger.error("Execution stoped")
            raise RuntimeError("User stoped execution.")
        self.logger.info("Execution continue")
