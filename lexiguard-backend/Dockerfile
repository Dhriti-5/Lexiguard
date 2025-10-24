# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Cloud Run will set PORT env variable)
EXPOSE 8080

# Set environment variable for Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
