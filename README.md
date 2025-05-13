# speechkit

Обертка в виде API над speech-to-text whisper-like моделью.

## Разработка

### Структура проекта

При разработке проекта придерживайтесь следующей структуры:

```bash
$ tree -L 3
.
├── conf  # project configuration with .env files
├── data  # models and uploaded audio files (if you will use local storage)
├── deploy  # helm charts for deployment and CI scripts
├── docker  # logging configuration and scripts for docker image (like start and entrypoint)
├── speechkit  # <- code goes here, not outside of this folder
│   ├── api  # implementation of handlers, middlewares, start script for FastAPI application
│   ├── broker  # implementation of tasks and message broker
│   ├── domain  # domain models and services / repositories (not implementation, just abstract classes)
│   └── infrastructure  # infrastructure aware implementations of repositories and services
├── tests  # <- tests go here, not near production code
│   ├── integration  # tests with real postgres and rabbitmq (up in containers)
│   ├── load  # handrunnable load tests with request fabrics and statics after run
│   └── unit  # unit tests (using mocks / in-memory / local filesystem only)
```

### Основные команды

Все команды описаны с пояснениями в Makefile, вызвать подсказку можно с помощью `make help`:
```bash
$ make help

-e Usage: make [target] ...

Help:
  help                Show this help
...
```

Придерживаемся подхода, что всё запускается внутри Docker-контейнеров:
- Контейнеры для приложения - api и worker (отдельно/ в одном контейнере)
- Контейнер для баз данных - postgres
- Контейнер для очереди сообщений - rabbitmq

Все вышеперечисленные контейнеры описаны в [docker-compose.yml](docker-compose.yml), который предназначен для локального запуска приложения и тестирования.
