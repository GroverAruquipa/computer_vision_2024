import time

from pynput import keyboard
from rich.progress import Progress, TaskID
from typing_extensions import override

from src.application.pipeline.pipeline_step import PipelineLoopBreaker
from src.domain.context import IterationNumberLoopConfig, KeypressLoopConfig, PipelineContext, TimerLoopConfig


class TimerLoopBreaker(PipelineLoopBreaker):
    def __init__(self, config: TimerLoopConfig):
        self.end_time: float = time.time() + config.duration
        self.progress: Progress = Progress()
        self.task_id: TaskID | None = None

    @override
    def should_break(self, context: PipelineContext):
        current_time = time.time()

        if self.task_id is None:
            self.task_id = self.progress.add_task("Processing...", total=self.end_time - (current_time - self.end_time))

        self.progress.update(self.task_id, completed=current_time)

        if current_time >= self.end_time:
            self.progress.stop()
            return True
        else:
            return False


class KeyPressLoopBreaker(PipelineLoopBreaker):
    def __init__(self, config: KeypressLoopConfig):
        self.config: KeypressLoopConfig = config
        self.is_key_pressed: bool = False
        self.is_called_once: bool = False
        self._listener: keyboard.Listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None):
        if key is None:
            return None
        if isinstance(key, keyboard.Key) and key.name == self.config.break_key:
            self.is_key_pressed = True
        if isinstance(key, keyboard.KeyCode) and key.char == self.config.break_key:
            self.is_key_pressed = True

    @override
    def should_break(self, context: PipelineContext):
        should_break = self.is_called_once and self.is_key_pressed
        if not self.is_called_once:
            self.is_called_once = True
            self.is_key_pressed = False
        return should_break

    def __del__(self):
        self._listener.stop()


class IterationNumberLoopBreaker(PipelineLoopBreaker):
    def __init__(self, config: IterationNumberLoopConfig):
        self.config: IterationNumberLoopConfig = config
        self.iterations: int = 0

    @override
    def should_break(self, context: PipelineContext):
        self.iterations += 1
        return self.iterations > self.config.iterations
