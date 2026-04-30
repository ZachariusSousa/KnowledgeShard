FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV KS_ENABLE_MODEL=0

WORKDIR /app

COPY requirements.txt .
COPY requirements-ml.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
ARG INSTALL_ML=false
RUN if [ "$INSTALL_ML" = "true" ]; then python -m pip install --no-cache-dir -r requirements-ml.txt; fi

COPY . .

VOLUME ["/app/data", "/app/weights"]

EXPOSE 8080

CMD ["python", "-m", "knowledgeshard.server", "--host", "0.0.0.0", "--port", "8080"]
