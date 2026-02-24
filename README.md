🚀 Deepfake Detection Using Deep Learning
Dockerized Django + PyTorch + CUDA GPU Deployment
📌 Overview

This project is a Deepfake Detection Web Application built using Deep Learning and deployed using Docker with NVIDIA CUDA GPU support.

The system allows users to upload videos and performs deepfake detection using a CNN-based PyTorch model with GPU-accelerated inference.

The entire application is fully containerized for reproducible and scalable deployment.

🎯 Key Highlights

✅ Deep Learning-based Deepfake Detection

✅ PyTorch GPU Inference (CUDA 11.3)

✅ Django Web Application

✅ Gunicorn Production Server

✅ Fully Dockerized (NVIDIA Runtime)

✅ Volume-based Media & Static Handling

✅ Production-ready Container Deployment

🧠 Model Architecture

Convolutional Neural Network (CNN)

Frame extraction from video

Preprocessing pipeline

GPU-based batch inference

Binary classification (Real / Fake)

Framework: PyTorch

Inference Device: NVIDIA GPU (CUDA Enabled)

🏗 System Architecture

User → Django Web App → Video Upload → Frame Extraction →
Preprocessing → PyTorch Model → GPU Inference → Prediction Output

Containerized with:

NVIDIA CUDA Runtime

Gunicorn WSGI Server

Django Backend

🛠 Tech Stack
Layer	Technology
Backend	Django
Deep Learning	PyTorch
GPU Acceleration	CUDA 11.3
Deployment	Docker
Server	Gunicorn
OS Base	Ubuntu 20.04 (CUDA Runtime)
📦 Dockerized Deployment (GPU Enabled)
🔹 Prerequisites

Docker Desktop

NVIDIA GPU Drivers

NVIDIA Container Toolkit

WSL2 (Windows Users)

Verify GPU access:

docker run --rm --gpus all nvidia/cuda:11.3.1-base nvidia-smi
🚀 Run Using Docker (Recommended)
1️⃣ Pull Image
docker pull YOUR_DOCKERHUB_USERNAME/deepfake-detector:0.1
2️⃣ Run Container
docker run --rm --gpus all \
-v static_volume:/home/app/staticfiles/ \
-v media_volume:/app/uploaded_videos/ \
--name deepfake-detector \
-p 8000:8000 \
YOUR_DOCKERHUB_USERNAME/deepfake-detector:0.1
3️⃣ Access Application

Open browser:

http://localhost:8000
🐳 Build Locally (Optional)
docker build -t deepfake-detector:0.1 .

Run:

docker run --rm --gpus all -p 8000:8000 deepfake-detector:0.1
📊 Features

Upload video for detection

Automatic frame extraction

GPU-accelerated model inference

Real-time classification

Clean UI with prediction results

Containerized reproducible deployment

⚡ Performance Optimization

CUDA runtime image used for GPU efficiency

Torch installed with CUDA 11.3 support

Gunicorn used for production-grade serving

Static and media volumes mounted externally

Docker-based isolated environment

📁 Project Structure
deepfake-detection/
│
├── Dockerfile
├── requirements_docker.txt
├── manage.py
├── project_settings/
├── detection_app/
├── static/
├── templates/
└── README.md
🔐 Production Considerations

DEBUG disabled in production

Environment variables configurable

GPU runtime isolation

Reproducible container deployment

Scalable to cloud GPU instances (AWS / GCP / Azure)

📈 Future Improvements

Add REST API endpoints

Add asynchronous task queue (Celery)

Deploy to AWS EC2 GPU instance

Model quantization for faster inference

CI/CD pipeline integration

👨‍💻 Author

Milind
Deep Learning & Backend Developer
