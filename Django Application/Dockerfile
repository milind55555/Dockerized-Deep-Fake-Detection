# ==========================================
# CUDA 11.3 Runtime Base (Lightweight)
# ==========================================
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# ==========================================
# Environment Variables
# ==========================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# ==========================================
# Install System Dependencies
# ==========================================
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-dev \
    build-essential \
    gcc \
    libgcc1 \
    libstdc++6 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python default
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# ==========================================
# Copy Clean Requirements
# ==========================================
COPY requirements_docker.txt .

RUN python -m pip install --upgrade pip

# ==========================================
# Install Torch (Stable Method)
# ==========================================
RUN pip install --no-cache-dir \
    torch==1.10.2  \
    torchvision==0.11.3 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# ==========================================
# Install Remaining Requirements
# ==========================================
RUN pip install --no-cache-dir -r requirements_docker.txt

# ==========================================
# Copy Project
# ==========================================
COPY . .

# ==========================================
# Collect Static
# ==========================================
RUN python manage.py collectstatic --noinput

EXPOSE 8000

# ==========================================
# Start Gunicorn
# ==========================================
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "project_settings.wsgi:application"]