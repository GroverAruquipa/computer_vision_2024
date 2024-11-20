import time

import cv2
from rich.progress import Progress, TaskID

from src.application.pipeline.pipeline_step import PipelineLoopBreaker
from src.domain.context import PipelineContext


class TimerLoopBreaker(PipelineLoopBreaker):
    def __init__(self, seconds: int):
        self.end_time = time.time() + seconds
        self.progress = Progress()
        self.task_id = None

    def should_break(self, context: PipelineContext):
        current_time = time.time()

        if self.task_id is None:
            self.task_id: TaskID = self.progress.add_task(
                "Processing...", total=self.end_time - (current_time - self.end_time)
            )

        self.progress.update(self.task_id, completed=current_time)

        if current_time >= self.end_time:
            self.progress.stop()
            return True
        else:
            return False


class KeyPressLoopBreaker(PipelineLoopBreaker):
    def __init__(self, key: int = 27):  # 27 is ESC
        self.break_key = key

    def todo(self, context: PipelineContext):
        key = cv2.waitKey(1)
        return key == self.break_key
