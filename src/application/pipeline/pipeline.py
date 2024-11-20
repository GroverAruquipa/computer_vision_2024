from src.application.pipeline.pipeline_step import PipelineStep
from src.domain.context import PipelineContext


class Pipeline:
    def __init__(self, pipeline_steps: list[PipelineStep], context: PipelineContext):
        self.pipelineSteps = pipeline_steps
        self.context = context

    def execute(self):
        for step in self.pipelineSteps:
            input_context = self.context
            output_context = step.execute(input_context)

            self.context = output_context
