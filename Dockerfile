FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy source code and models
COPY src/ ./src/
COPY models/ ./models/
COPY api.py .

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
