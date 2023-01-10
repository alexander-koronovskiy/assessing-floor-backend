FROM python:3.10-slim
ENV PYTHONPATH="/usr/src:/usr/src/app"

ENV GUNICORN_TIMEOUT="240"
ENV GUNICORN_WORKERS="4"

ENV ASGI_APP="main:create_app()"
ENV ASGI_APP_DIR="/usr/src/app"
ENV ASGI_CONFIG_PATH="config.yml"
ENV ASGI_WORKER="uvicorn.workers.UvicornWorker"

RUN apt-get update && apt-get install -y --no-install-recommends build-essential wget
COPY . "${ASGI_APP_DIR}"
WORKDIR "${ASGI_APP_DIR}"

RUN mkdir -p "${ASGI_APP_DIR}"
COPY requirements.txt "${ASGI_APP_DIR}/requirements.txt"
RUN pip install -r "${ASGI_APP_DIR}/requirements.txt"

EXPOSE 8000
CMD gunicorn \
    -b ":8000" \
    -k "${ASGI_WORKER}" \
    -w "${GUNICORN_WORKERS}" \
    --timeout "${GUNICORN_TIMEOUT}" \
    --access-logfile - \
    --error-logfile - \
    "${ASGI_APP}"
