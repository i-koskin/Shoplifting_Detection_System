FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies including FFmpeg with RTSP support
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libavcodec-extra \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    curl \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Download YOLO model (using your actual model)
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11s-pose.pt')"

# Copy application code
COPY . .

# Create directory structure (only essential directories that won't be mounted as volumes)
RUN mkdir -p test_videos

# Set proper permissions
RUN chmod -R 755 /app/test_videos

# Verify installations
RUN python3 --version && \
    pip3 --version && \
    ffmpeg -version | head -n1

# Verify FFmpeg codecs and protocols
RUN echo "=== FFmpeg H.264 support ===" && \
    ffmpeg -codecs | grep h264 && \
    echo "=== FFmpeg RTSP support ===" && \
    ffmpeg -protocols | grep rtsp

# Add a simple health check script
RUN echo '#!/bin/bash\n\
# Simple health check that verifies Python and key imports work\n\
python3 -c "import cv2, torch, pandas; print(\"Health check: OK\")"\n\
exit $?' > /healthcheck.sh && chmod +x /healthcheck.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /healthcheck.sh

# Run the application
CMD ["python3", "shoplifting_detector.py"]