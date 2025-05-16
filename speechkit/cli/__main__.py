from speechkit.cli.commands.download_model import download_model_command  # noqa: F401
from speechkit.cli.commands.root import cli
from speechkit.cli.commands.upload_file import upload_file_command  # noqa: F401
from speechkit.infrastructure import logs


if __name__ == '__main__':
    logs.configure_logging()
    cli()
