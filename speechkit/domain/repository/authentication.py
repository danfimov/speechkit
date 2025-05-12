import abc


class AbstractAuthenticationRepository(abc.ABC):
    @abc.abstractmethod
    async def check_credentials(self, service_name: str, token: str) -> bool: ...
