import pathlib

import click
import structlog
import uvloop

from speechkit import dependencies
from speechkit.cli.commands.root import cli
from speechkit.domain.service import model_downloader


logger = structlog.get_logger('speechkit.cli')


async def download_and_save_model(model_name: str, save_path: str) -> None:
    service = await dependencies.container.get(model_downloader.AbstractModelDownloaderService)
    try:
        await service.act(
            destination_path=pathlib.Path(save_path),
            model_name=model_name,
        )
    except Exception:
        logger.exception('Error occurred during model downloading')
    finally:
        await dependencies.container.close()


@cli.command('download-model')
@click.option(
    '--model_name',
    default='antony66/whisper-large-v3-russian',
    help='Name of the model to download from Hugging Face',
)
@click.option('--save_path', default='data/whisper-model', help='Path to save the downloaded model')
def download_model_command(model_name: str, save_path: str) -> None:
    """Download a model from Hugging Face or S3 and save it to the specified path."""
    uvloop.run(
        download_and_save_model(model_name, save_path),
    )
