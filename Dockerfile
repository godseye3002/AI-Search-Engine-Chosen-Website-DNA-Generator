FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Removed the apt-get install for libgl1 since OpenCV is gone.
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["sh", "-c", "uvicorn serverless_api:app --host 0.0.0.0 --port ${PORT:-8000}"]