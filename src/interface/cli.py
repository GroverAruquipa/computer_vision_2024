import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from src.application.pipeline.pipeline_builder import PipelineBuilder
from src.domain.context import PipelineConfig

app = typer.Typer()
logger = logging.getLogger(__name__)


def parse_config(config_path: Path) -> PipelineConfig:
    with open(config_path) as f:
        config_dict = json.load(f)

    try:
        pipeline_config = PipelineConfig.parse_obj(config_dict)

    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise typer.Exit(1) from e

    return pipeline_config


default_config_path = Path("./config/config.json")


@app.command()
def main(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=False,
            help="Path to configuration file",
        ),
    ] = default_config_path,
):
    pipeline_config = parse_config(config_path)

    try:
        pipelineBuilder = PipelineBuilder(pipeline_config.context)
        [pipelineBuilder.create_step(step_config) for step_config in pipeline_config.steps]

        pipelineBuilder.build().execute()

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    main()
