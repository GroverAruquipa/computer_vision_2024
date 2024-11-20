import logging
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from src.application.pipeline.pipeline_builder import PipelineBuilder
from src.domain.context import PipelineConfig

app = typer.Typer()
logger = logging.getLogger(__name__)


def parse_config(config_path: Path) -> PipelineConfig:
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    try:
        pipeline_config = PipelineConfig.parse_obj(config_dict)

    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise typer.Exit(1) from e

    return pipeline_config


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
    ] = "./config/config.yaml",
):
    pipeline_config = parse_config(config_path)

    try:
        pipeline = PipelineBuilder(pipeline_config).build()
        pipeline.execute()

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    main()


#
# def set_capture_strategy(source, video_path) -> CaptureStrategy:
#     if source == "kinect":
#         from src.domain.capture.kinnect import KinectCaptureStrategy
#
#         return KinectCaptureStrategy()
#     elif source == "webcam":
#         from src.domain.capture.webcam import WebcamCaptureStrategy
#
#         return WebcamCaptureStrategy()
#     else:
#         from src.domain.capture.video import VideoFileCaptureStrategy
#
#         return VideoFileCaptureStrategy(video_path)
#
#
# def validate_cli_params(config_path: str, source: str, video_path: str):
#     if not Path.exists(Path(config_path)):
#         raise click.BadParameter("Config path must be a valid yaml file path")
#
#     if source == "kinect" and os.name != "nt":
#         raise click.BadParameter("Kinect is only functional on Windows")
#
#     if source == "video" and not Path.exists(Path(video_path)):
#         raise click.BadParameter("Video path must be provided when using video source")
#
#
# def load_config(config_path) -> PipelineContext:
#     with open(config_path) as f:
#         data = yaml.safe_load(f)
#         return from_dict(data_class=PipelineContext, data=data["config"])
#


# @click.command()
# @click.option(
#     "--config-path",
#     type=click.Path(exists=True),
#     default=Path("./config/default_config.yaml"),
#     help="Path to configuration file",
# )
# @click.option(
#     "--source",
#     type=click.Choice(["kinect", "webcam", "video"]),
#     default="webcam",
#     help="Video source to use",
# )
# @click.option(
#     "--video-path",
#     type=click.Path(exists=True),
#     help="Path to video file if using video source",
# )
