
FROM python:3.10-slim
WORKDIR /app
COPY artifacts/ /app/artifacts/
COPY app/app.py /app/app.py
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn pandas joblib numpy
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
