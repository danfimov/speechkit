import enum
import http
import typing as tp

import fastapi
import pydantic
from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic

from speechkit.domain.service import recognition_service


router = fastapi.APIRouter(tags=['Recognition'], prefix='/recognition', route_class=DishkaRoute)
security = HTTPBasic()


class RecognitionModel(enum.StrEnum):
    WHISPER = 'whisper'


class RecognitionResponseOk(pydantic.BaseModel):
    text: str


class RecognitionResponseError(pydantic.BaseModel):
    error: str


@router.post(
    '/',
    response_model=None,
    responses={
        200: {'model': RecognitionResponseOk},
        400: {'model': RecognitionResponseError},
    },
)
async def recognition(
    file: tp.Annotated[fastapi.UploadFile, fastapi.File(...)],
    model_name: tp.Annotated[RecognitionModel, fastapi.Form()],
    service: FromDishka[recognition_service.AbstractRecognitionService],
    # TODO: add auth middleware and then add credentials here
    # credentials: tp.Annotated[HTTPBasicCredentials, fastapi.Depends(security)],
) -> JSONResponse:
    try:
        recognised_text = await service.act(
            model_name=model_name,
            audio_file_content=await file.read(),
            audio_file_name=file.filename,
            audio_file_content_type=file.content_type,
        )
    except recognition_service.RecognitionServiceError:
        return JSONResponse(status_code=http.HTTPStatus.BAD_REQUEST, content={'error': 'Recognition error'})
    return JSONResponse(status_code=http.HTTPStatus.OK, content={'text': recognised_text})
