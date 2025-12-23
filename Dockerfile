FROM python:3.10-slim

WORKDIR /app

# Copy project
COPY api /app/api
COPY models /app/models
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
