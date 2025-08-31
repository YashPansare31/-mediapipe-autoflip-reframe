FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-dri  \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Create necessary directories
RUN mkdir -p /app/data /app/output /app/temp

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -m -u 1000 reframer && \
    chown -R reframer:reframer /app
USER reframer

# Expose port (if needed for future web interface)
EXPOSE 8080

# Default command
CMD ["python", "-m", "src.cli", "--help"]