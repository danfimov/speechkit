#!/bin/sh

set -e

if [ -f "/var/run/secrets/app/.env" ]; then
    echo "Loading environment variables from /var/run/secrets/app/.env"
    export $(grep -v '^#' /var/run/secrets/app/.env | xargs)
else
    echo "Warning: /var/run/secrets/app/.env file not found, skipping env variables loading"
fi

if [ "$APP_NAME" = "api" ]; then
    echo "Starting API server"
    uvicorn speechkit.api.__main__:get_app --host "" --port 8000 --factory \
        --log-config=docker/json-logging.yaml --proxy-headers --forwarded-allow-ips=* --timeout-keep-alive=60
elif [ "$APP_NAME" = "worker" ]; then
    echo "Starting taskiq worker"
    taskiq worker speechkit.broker:broker speechkit.broker.tasks --workers 1 --shutdown-timeout 10
elif [ "$APP_NAME" = "migrator" ]; then
    echo "Applying migrations"
    python3 -m speechkit.infrastructure.database.migrations upgrade head
else
    echo "Unknown app name: $APP_NAME"
    exit 1
fi

exec "$@"
