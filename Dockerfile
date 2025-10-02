# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# First install all standard dependencies except qarray
RUN grep -v "^qarray$" requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt

# Install qarray separately 
# Option 1: If qarray is available on PyPI
RUN pip install --no-cache-dir qarray || echo "qarray not found on PyPI, may need custom installation"

# Option 2: If you need to install from GitHub (uncomment and update as needed)
# RUN pip install --no-cache-dir git+ssh://git@github.com/b-vanstraaten/qarray-latched.git@c076d4cef57a071dd6e52458ad5937589747c18f

# Set up Ray/RLlib environment variables for better performance
ENV RAY_DEDUP_LOGS=0
ENV OMP_NUM_THREADS=1

# Copy the entire project
COPY . .

# Set the default command (can be overridden when running the container)
CMD ["python", "src/swarm/training/train.py"]
