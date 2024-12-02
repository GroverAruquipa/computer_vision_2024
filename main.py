import logging

import typer
from rich.logging import RichHandler
from rich.traceback import install

install(show_locals=True)
logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])


if __name__ == "__main__":
    from src.interface.cli import main

    typer.run(main)
