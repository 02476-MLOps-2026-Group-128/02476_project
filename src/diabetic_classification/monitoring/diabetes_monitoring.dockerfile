FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi nltk evidently google-cloud-storage --no-cache-dir

COPY src/diabetic_classification/monitoring/diabetes_monitoring.py .

EXPOSE $PORT

CMD exec uvicorn diabetes_monitoring:app --port $PORT --host 0.0.0.0 --workers 1