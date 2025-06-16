# Use the official Python 3.12 slim image
FROM python:3.12.9-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gdal-bin libgdal-dev && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy & install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default entrypoint; user supplies script and args
ENTRYPOINT ["python", "-u"]
CMD ["--help"]