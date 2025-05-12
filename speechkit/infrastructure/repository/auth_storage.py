from speechkit.domain.repository import authentication
from speechkit.infrastructure.database import session_provider


class PostgresAuthenticationRepository(authentication.AbstractAuthenticationRepository):
    def __init__(self, session_provider: session_provider.AsyncPostgresSessionProvider) -> None:
        self._session_provider = session_provider

    async def check_credentials(self, service_name: str, token: str) -> bool:  # noqa: ARG002
        return False
