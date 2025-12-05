# Multi-stage build for VAM Tools
# Use NVIDIA CUDA base image for GPU acceleration
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 as base

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # ExifTool for metadata extraction
    libimage-exiftool-perl \
    # Image processing libraries
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libheif-dev \
    # Video processing
    ffmpeg \
    # Build tools
    gcc \
    g++ \
    make \
    curl \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml README.md ./

# Install Python dependencies (base)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Install GPU acceleration packages (PyTorch with CUDA support)
# Using CUDA 12.4 compatible wheels
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Copy application code
COPY vam_tools/ ./vam_tools/

# Create directories for catalogs and photos
RUN mkdir -p /app/catalogs /app/photos

# Expose port for web API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api || exit 1

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "vam_tools.web.api:app", "--host", "0.0.0.0", "--port", "8000"]
