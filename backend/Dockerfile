FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && JAVA_BIN="$(readlink -f "$(command -v java)")" \
    && JAVA_HOME_DIR="$(dirname "$(dirname "$JAVA_BIN")")" \
    && ln -s "$JAVA_HOME_DIR" /opt/java-home \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/opt/java-home

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY backend ./backend

EXPOSE 5001

WORKDIR /app/backend

CMD ["python", "run.py"]
