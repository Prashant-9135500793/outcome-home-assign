FROM python:3.11-slim

# Install system dependencies (poppler for pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and schemas
COPY app/ ./app/
COPY sample/schemas/ ./sample/schemas/

# Expose port
EXPOSE 8080

# Environment defaults
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV LLM_BASE_URL=https://openrouter.ai/api/v1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
