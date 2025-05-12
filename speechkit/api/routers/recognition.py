import enum
import http
import typing as tp

import fastapi
import pydantic
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from speechkit import dependencies
from speechkit.domain.service import recognition_service


router = fastapi.APIRouter(tags=['Recognition'], prefix='/recognition')
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
    service: tp.Annotated[
        recognition_service.AbstractRecognitionService,
        fastapi.Depends(dependencies.get_recognition_service),
    ],
    credentials: tp.Annotated[HTTPBasicCredentials, fastapi.Depends(security)],  # noqa: ARG001
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
