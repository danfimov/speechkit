import fastapi
from fastapi import responses


router = fastapi.APIRouter(tags=['System'])


@router.get('/', summary='Главная страница, перенаправляет на Swagger UI')
async def root() -> responses.RedirectResponse:
    return responses.RedirectResponse(url='/docs')


@router.get('/liveness', name='liveness', summary='Проверка работоспособности сервиса')
async def get_liveness() -> dict[str, str]:
    return {'status': 'OK'}


@router.get('/readiness', name='readiness', summary='Проверка готовности обслуживать входящие запросы')
async def get_readiness() -> dict[str, str]:
    # TODO: maybe add "select 1" to database
    return {'status': 'OK'}
