# type: ignore  # noqa: PGH003

import base64
import hashlib

import fastapi

from speechkit.domain.repository import authentication


class AuthMiddleware:
    def __init__(
        self,
        app: fastapi.FastAPI,
        auth_repository: authentication.AbstractAuthenticationRepository,
        public_paths: list[str] | None = None,
    ) -> None:
        if public_paths is None:
            public_paths = ['/docs']
        self.app = app
        self.auth_repository = auth_repository
        self.public_paths = public_paths or ['/public']

    async def __call__(  # noqa: ANN204, PLR0911
        self,
        scope,  # noqa: ANN001
        receive,  # noqa: ANN001
        send,  # noqa: ANN001
    ):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)

        path = scope['path']
        if any(path.startswith(public_path) for public_path in self.public_paths):
            return await self.app(scope, receive, send)

        headers = dict(scope['headers'])
        auth_header = headers.get(b'authorization', b'').decode()

        if not auth_header:
            return await self.unauthorized_response(scope, receive, send)

        try:
            auth_type, auth_info = auth_header.split(' ', 1)
            if auth_type.lower() != 'basic':
                return await self.app(scope, receive, send)

            service_name, token = base64.b64decode(auth_info).decode().split(':')
            hashed_token = hashlib.sha256(token.encode()).hexdigest()

            is_valid = await self.auth_repository.check_credentials(service_name, hashed_token)
            if not is_valid:
                return await self.unauthorized_response(scope, receive, send)

            scope['service_name'] = service_name
            return await self.app(scope, receive, send)

        except Exception:  # noqa: BLE001
            return await self.unauthorized_response(scope, receive, send)

    async def unauthorized_response(
        self,
        scope,  # noqa: ANN001
        receive,  # noqa: ANN001
        send,  # noqa: ANN001
    ) -> None:
        response = fastapi.Response(
            status_code=401,
            headers={'WWW-Authenticate': "Basic realm='Authentication Required'"},
        )
        await response(scope, receive, send)
