from src.application.pipeline.pipeline_step import PipelineStep
from src.domain.context import PipelineContext


class Pipeline:
    def __init__(self, pipeline_steps: list[PipelineStep], context: PipelineContext):
        self.pipeline_steps: list[PipelineStep] = pipeline_steps
        self.context: PipelineContext = context

    def execute(self):
        try:
            for step in self.pipeline_steps:
                input_context = self.context
                output_context = step.execute(input_context)
                self.context = output_context

        except Exception as e:
            for step in self.pipeline_steps:
                step.cleanup()
            raise e
