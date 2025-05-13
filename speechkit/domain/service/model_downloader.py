import abc
import pathlib


class ModelDownloaderError(Exception):
    pass


class AbstractModelDownloaderService:
    @abc.abstractmethod
    async def act(self, model_name: str, destination_path: pathlib.Path) -> None: ...

    async def close(self) -> None:
        """Метод для закрытия подключений, выполняется при завершении работы сервиса."""
