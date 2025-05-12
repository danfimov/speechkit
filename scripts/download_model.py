import logging
import pathlib

import click
import uvloop
from tochka_infrastructure import logs

from speechkit import dependencies


logger = logging.getLogger(__name__)


async def download_and_save_model(model_name: str, save_path: str) -> None:
    settings = dependencies.get_settings()
    logs.configure_logging(settings.log_config_path, settings.log_level)

    service = dependencies.get_model_downloader()
    try:
        await service.act(
            destination_path=pathlib.Path(save_path),
            model_name=model_name,
        )
    except Exception:
        logger.exception('Error occurred during model downloading')
    finally:
        await dependencies.stop_application()


@click.command()
@click.option(
    '--model_name',
    default='antony66/whisper-large-v3-russian',
    help='Name of the model to download from Hugging Face',
)
@click.option('--save_path', default='data/whisper-model', help='Path to save the downloaded model')
def command(model_name: str, save_path: str) -> None:
    """Download a model from Hugging Face or S3 and save it to the specified path."""
    uvloop.run(
        download_and_save_model(model_name, save_path),
    )


if __name__ == '__main__':
    command()
