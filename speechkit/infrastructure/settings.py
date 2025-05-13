import os
import typing as tp
from urllib.parse import quote, urlparse

import pydantic
import pydantic_settings


def _remove_prefix(value: str, prefix: str) -> str:
    if value.startswith(prefix):
        return value[len(prefix) :]
    return value


class PostgresSettings(pydantic.BaseModel):
    """Настройки для подключения к PostgreSQL."""

    driver: str = 'postgresql'
    host: str = 'pg-public-stage.query.consul'
    port: int = 5432
    user: str
    password: pydantic.SecretStr
    database: str

    min_pool_size: int
    max_pool_size: int

    @property
    def dsn(self) -> pydantic.SecretStr:
        return pydantic.SecretStr(
            f'{self.driver}://{self.user}:{quote(self.password.get_secret_value())}@{self.host}:{self.port}/{self.database}',
        )

    @pydantic.model_validator(mode='before')
    @classmethod
    def __parse_dsn(cls, values: dict[str, tp.Any]) -> dict[str, tp.Any]:
        dsn = values.get('dsn')
        if dsn is not None and not isinstance(dsn, str):
            msg = "Field 'dsn' must be str"
            raise TypeError(msg)
        if not dsn:
            return values
        parsed_dsn = urlparse(dsn)
        values['driver'] = parsed_dsn.scheme
        values['host'] = parsed_dsn.hostname
        values['port'] = parsed_dsn.port
        values['user'] = parsed_dsn.username
        values['password'] = parsed_dsn.password
        values['database'] = _remove_prefix(parsed_dsn.path, '/')
        return values



class SentrySettings(pydantic.BaseModel):
    """Настройки подключения к Sentry."""

    dsn: pydantic.AnyHttpUrl
    logging_level: str = 'INFO'
    logging_event_level: str = 'ERROR'
    sample_rate: float = 1
    traces_sample_rate: float = 0


class TracingSettings(pydantic.BaseModel):
    """Настройки для трассировки."""

    otlp_sampling_ratio: float = pydantic.Field(
        default=1.0,
        description='Ratio for sampling traces, `0` is for disabled traces, `1` is for 100%',
    )
    otlp_enable_console_exporter: bool = pydantic.Field(default=False)
    otlp_grpc_endpoint: str | None = pydantic.Field(default=None)


class AMQPSettings(pydantic.BaseModel):
    scheme: tp.Literal['amqp', 'amqps'] = 'amqps'
    host: str
    port: int = 5671
    user: str
    password: str
    management_port: int = 15672
    stomp_port: int = 61613
    vhost: str

    def get_url(self) -> str:
        return f'{self.scheme}://{self.user}:{self.password}@{self.host}:{self.port}/{self.vhost}'


class S3Settings(pydantic.BaseModel):
    access_key: str
    secret_key: str
    region: str = 'us-east-1'
    endpoint_url: str = 'https://s3-proxy.query.consul'


class Settings(pydantic_settings.BaseSettings):
    # Service-agnostic settings
    instance_id: str | None = pydantic.Field(
        default=None,
        validation_alias=pydantic.AliasChoices('HOSTNAME', 'INSTANCE_ID'),
        description='HOSTNAME есть в переменных окружения в каждом pod в k8s',
    )
    datacenter: str | None = pydantic.Field(
        default=None,
        validation_alias=pydantic.AliasChoices('CONSUL_DATACENTER', 'PHYSICAL_DATACENTER'),
        description='CONSUL_DATACENTER есть в переменных окружения в каждом pod в k8s',
    )
    service_name: str = pydantic.Field(
        default='unknown',
        validation_alias=pydantic.AliasChoices('NAMESPACE', 'SERVICE_NAME'),
        description='NAMESPACE есть в переменных окружения в каждом pod в k8s',
    )
    app_version: str = pydantic.Field(
        default='unknown',
        validation_alias=pydantic.AliasChoices('VERSION', 'APP_VERSION'),
        description='Желательно зашивать VERSION в docker image в качестве аргумента при сборке',
    )
    environment: str = pydantic.Field(
        default='unknown',
        validation_alias=pydantic.AliasChoices('ENVIRONMENT', 'SENTRY_ENVIRONMENT'),
        description='ENVIRONMENT есть в переменных окружения в каждом pod в k8s',
    )
    log_level: str = pydantic.Field(
        default='INFO',
        description='Желательно не использовать DEBUG в логах, кроме случаев локальной отладки',
    )
    metrics_port: int | None = None
    sentry: SentrySettings | None = None
    tracing: TracingSettings = TracingSettings()

    # Service settings

    app_name: tp.Literal['api', 'worker', 'migrator'] = pydantic.Field(
        default='api',
        description='Defines that part of service is working',
    )

    # Storages
    amqp: AMQPSettings | None = None
    postgres: PostgresSettings
    s3: S3Settings

    broker_type: tp.Literal['amqp', 'memory'] = 'amqp'
    downloader_type: tp.Literal['s3', 'huggingface'] = 's3'

    s3_bucket_models: str = 'speechkit-models'
    s3_bucket_files: str = 'speechkit-files'

    model_name: str = 'antony66/whisper-large-v3-russian'

    # Inference
    inference_type: tp.Literal['pytorch', 'onnx', 'vllm'] = 'onnx'
    intra_op_num_threads: int | None = None
    inter_op_num_threads: int | None = None

    # feature flags
    enable_auth: bool = False

    model_config = pydantic_settings.SettingsConfigDict(
        env_nested_delimiter='__',
        env_file=('conf/.env', os.getenv('ENV_FILE', '.env')),
        env_file_encoding='utf-8',
    )
