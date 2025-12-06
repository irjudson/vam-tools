# Multi-stage build for VAM Tools
# Use NVIDIA CUDA base image for GPU acceleration
# Using CUDA 12.6 for RTX 5060 Ti Blackwell (sm_120) support
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 as base

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
# Using nightly build with CUDA 12.8 for RTX 5060 Ti Blackwell (sm_120) support
# CUDA 12.8+ is required for Blackwell architecture (sm_120)
RUN pip install --no-cache-dir --pre \
    torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install tagging dependencies (OpenCLIP and Ollama client)
RUN pip install --no-cache-dir \
    open-clip-torch>=2.24.0 \
    ftfy>=6.1.0 \
    ollama>=0.3.0

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
