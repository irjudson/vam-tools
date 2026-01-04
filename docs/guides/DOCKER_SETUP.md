# Lumina - Docker Quick Start

Run Lumina with Docker for easy deployment with GPU-accelerated background processing.

## 🚀 Quick Start

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit paths in .env
nano .env
# Set:
#   CATALOG_PATH=/path/to/catalogs
#   PHOTOS_PATH=/path/to/photos

# 3. Build and start
docker-compose up -d

# 4. Access web interface
open http://localhost:8000
```

## 📦 What's Included

- **Web UI** (port 8000) - Catalog browsing and job submission
- **Celery Worker** - Background processing with GPU support
- **Redis** - Message broker
- **Flower** (optional, port 5555) - Job monitoring

## 🎯 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| Catalog Viewer | http://localhost:8000 | Browse photos |
| Job Management | http://localhost:8000/static/jobs.html | Submit/monitor jobs |
| API Docs | http://localhost:8000/docs | Interactive API |
| Flower | http://localhost:5555 | Celery monitoring (optional) |

## 🔧 Common Commands

```bash
# View logs
docker-compose logs -f celery-worker

# Scale workers
docker-compose up -d --scale celery-worker=4

# Restart service
docker-compose restart web

# Stop all
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## 🎮 GPU Support

Automatically enabled if NVIDIA GPU + drivers are available.

**Verify GPU access**:
```bash
docker exec vam-celery-worker nvidia-smi
```

## 📚 Full Documentation

See [docs/DOCKER_DEPLOYMENT.md](docs/DOCKER_DEPLOYMENT.md) for complete guide including:
- Detailed configuration
- GPU troubleshooting
- Production deployment
- Performance tuning
- Monitoring and logging

## 🐛 Troubleshooting

**Jobs not processing?**
```bash
docker-compose logs celery-worker
docker-compose restart celery-worker
```

**Can't connect?**
```bash
docker-compose ps  # Check all services are running
docker-compose up -d  # Start if needed
```

**Permission errors?**
```bash
sudo chown -R $USER:$USER /path/to/catalogs
```

## 📖 Next Steps

1. **Submit a job**: Open http://localhost:8000/static/jobs.html
2. **Browse catalog**: http://localhost:8000
3. **Read full docs**: [DOCKER_DEPLOYMENT.md](docs/DOCKER_DEPLOYMENT.md)
# Lumina - Docker Deployment Guide

Complete guide for deploying Lumina with Docker, Celery, and Redis for background job processing with GPU acceleration.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Services](#services)
- [GPU Support](#gpu-support)
- [Usage](#usage)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Overview

Lumina provides a complete Docker-based deployment with:

- **Web API** - FastAPI server for catalog browsing and job submission
- **Celery Workers** - Background processing for analysis, organization, and thumbnails
- **Redis** - Message broker and result backend
- **Flower** - Optional Celery monitoring UI
- **GPU Support** - CUDA acceleration for perceptual hashing and analysis

## Prerequisites

### Required

- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- At least 4GB free RAM
- 10GB free disk space

### For GPU Support

- **NVIDIA GPU** with compute capability >= 3.5
- **NVIDIA Container Toolkit** installed
- **CUDA** 12.4+ compatible driver

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/irjudson/lumina.git
cd lumina
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your paths
nano .env
```

Update these critical paths:
```bash
CATALOG_PATH=/path/to/your/catalogs    # Where catalogs are stored
PHOTOS_PATH=/path/to/your/photos       # Your photo library (read-only)
```

### 3. Build and Start Services

```bash
# Build Docker images
docker-compose build

# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

### 4. Access Web Interface

- **Catalog Viewer**: http://localhost:8000
- **Job Management**: http://localhost:8000/static/jobs.html
- **API Docs**: http://localhost:8000/docs
- **Flower (optional)**: http://localhost:5555

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CATALOG_PATH` | `./catalogs` | Host path to catalog directory |
| `PHOTOS_PATH` | `./photos` | Host path to photos (mounted read-only) |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `CELERY_BROKER_URL` | `redis://redis:6379/0` | Redis broker URL |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/0` | Redis result backend URL |

### Docker Compose Profiles

```bash
# Start with Flower monitoring
docker-compose --profile monitoring up -d

# Start without GPU
docker-compose up -d web celery-worker redis  # Remove deploy.resources section first
```

---

## Services

### Web (FastAPI)

- **Port**: 8000
- **Function**: REST API and web UI
- **Resources**: 1 CPU, 2GB RAM
- **GPU**: Optional (for on-demand analysis)

### Celery Worker

- **Function**: Background job processing
- **Resources**: 2 CPUs, 4GB RAM
- **GPU**: Yes (for analysis tasks)
- **Concurrency**: 2 workers

Scale workers:
```bash
docker-compose up -d --scale celery-worker=4
```

### Redis

- **Port**: 6379
- **Function**: Message broker and result backend
- **Resources**: 512MB RAM
- **Data**: Persisted in `redis-data` volume

### Flower (Optional)

- **Port**: 5555
- **Function**: Celery monitoring dashboard
- **Enable**: `docker-compose --profile monitoring up -d`

---

## GPU Support

### Verify GPU Access

```bash
# Check GPU in web container
docker exec vam-web nvidia-smi

# Check GPU in worker container
docker exec vam-celery-worker nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.4    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... Off  | 00000000:01:00.0  On |                  N/A |
|  0%   45C    P8    15W / 250W |    500MiB / 11264MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### GPU Troubleshooting

If GPU is not detected:

1. **Check NVIDIA drivers**:
   ```bash
   nvidia-smi  # On host machine
   ```

2. **Verify NVIDIA runtime**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

3. **Check Docker daemon config** (`/etc/docker/daemon.json`):
   ```json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     },
     "default-runtime": "nvidia"
   }
   ```

4. **Restart Docker**:
   ```bash
   sudo systemctl restart docker
   docker-compose restart
   ```

---

## Usage

### Submit Analysis Job (Web UI)

1. Open http://localhost:8000/static/jobs.html
2. Go to "Submit Jobs" tab
3. Fill in "Analyze Catalog" form:
   - Catalog Path: `/app/catalogs/my-catalog`
   - Source Directories: `/app/photos`
   - Detect Duplicates: ✓ (if desired)
4. Click "Submit Analysis Job"
5. Switch to "Monitor Jobs" tab to watch progress

### Submit Analysis Job (CLI)

```bash
# Using curl
curl -X POST http://localhost:8000/api/jobs/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "catalog_path": "/app/catalogs/my-catalog",
    "source_directories": ["/app/photos"],
    "detect_duplicates": true,
    "similarity_threshold": 5
  }'

# Response
{
  "job_id": "abc123-def456-...",
  "status": "PENDING",
  "message": "Analysis job submitted successfully"
}
```

### Monitor Job Progress

```bash
# Get job status
curl http://localhost:8000/api/jobs/{job_id}

# Stream progress with SSE
curl -N http://localhost:8000/api/jobs/{job_id}/stream

# List all active jobs
curl http://localhost:8000/api/jobs
```

### Cancel Running Job

```bash
curl -X DELETE http://localhost:8000/api/jobs/{job_id}
```

### Organize Catalog

```bash
curl -X POST http://localhost:8000/api/jobs/organize \
  -H "Content-Type: application/json" \
  -d '{
    "catalog_path": "/app/catalogs/my-catalog",
    "output_directory": "/app/organized",
    "operation": "copy",
    "directory_structure": "YYYY-MM",
    "dry_run": true
  }'
```

### Generate Thumbnails

```bash
curl -X POST http://localhost:8000/api/jobs/thumbnails \
  -H "Content-Type: application/json" \
  -d '{
    "catalog_path": "/app/catalogs/my-catalog",
    "sizes": [256, 512],
    "quality": 85,
    "force": false
  }'
```

---

## Monitoring

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f celery-worker
docker-compose logs -f web

# Last 100 lines
docker-compose logs --tail=100 celery-worker
```

### Celery Flower Dashboard

```bash
# Start Flower
docker-compose --profile monitoring up -d

# Open browser
open http://localhost:5555
```

Features:
- Real-time task monitoring
- Worker status and stats
- Task history and results
- Broker monitoring

### Redis Monitoring

```bash
# Connect to Redis CLI
docker exec -it vam-redis redis-cli

# Check queue length
LLEN celery

# Monitor commands
MONITOR
```

### Resource Usage

```bash
# All containers
docker stats

# Specific container
docker stats vam-celery-worker
```

---

## Troubleshooting

### Worker Not Processing Jobs

**Check worker status**:
```bash
docker-compose logs celery-worker
docker exec vam-celery-worker celery -A vam_tools.jobs.celery_app inspect active
```

**Common issues**:
- Redis not reachable → Check `docker-compose ps redis`
- Worker crashed → Check logs for errors
- Out of memory → Reduce concurrency or add RAM

### Jobs Stuck in PENDING

**Check broker connection**:
```bash
docker exec vam-celery-worker python -c "from lumina.jobs.celery_app import app; print(app.connection().ensure_connection())"
```

**Restart worker**:
```bash
docker-compose restart celery-worker
```

### GPU Not Used During Analysis

**Verify GPU in worker**:
```bash
docker exec vam-celery-worker nvidia-smi
```

**Check PyTorch GPU**:
```bash
docker exec vam-celery-worker python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory

**Reduce worker concurrency**:
```yaml
# docker-compose.yml
command: celery -A vam_tools.jobs.celery_app worker --loglevel=info --concurrency=1
```

**Limit Docker memory**:
```yaml
services:
  celery-worker:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Permission Errors

**Fix catalog permissions**:
```bash
sudo chown -R $USER:$USER /path/to/catalogs
sudo chmod -R 755 /path/to/catalogs
```

**Fix photo permissions** (if not read-only):
```bash
sudo chmod -R 755 /path/to/photos
```

---

## Production Deployment

### Security Considerations

1. **Disable development features**:
   ```yaml
   # docker-compose.yml
   command: uvicorn vam_tools.web.api:app --host 0.0.0.0 --port 8000  # Remove --reload
   ```

2. **Use secrets for sensitive data**:
   ```yaml
   services:
     web:
       environment:
         - CELERY_BROKER_URL=${CELERY_BROKER_URL}  # From .env, not hardcoded
   ```

3. **Set up reverse proxy** (nginx):
   ```nginx
   server {
       listen 80;
       server_name vam.example.com;

       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

4. **Enable HTTPS** with Let's Encrypt

5. **Restrict CORS origins**:
   ```bash
   Lumina_CORS_ORIGINS=https://vam.example.com
   ```

### Performance Tuning

1. **Scale workers based on CPU cores**:
   ```bash
   docker-compose up -d --scale celery-worker=8
   ```

2. **Optimize Redis**:
   ```yaml
   redis:
     command: redis-server --maxmemory 1gb --maxmemory-policy allkeys-lru
   ```

3. **Use SSD for catalog storage**

4. **Enable Redis persistence**:
   ```yaml
   redis:
     command: redis-server --appendonly yes
   ```

### Backup and Recovery

**Backup catalogs**:
```bash
# Create backup
tar -czf catalog-backup-$(date +%Y%m%d).tar.gz $CATALOG_PATH

# Backup Redis data
docker exec vam-redis redis-cli BGSAVE
```

**Restore catalog**:
```bash
tar -xzf catalog-backup-20250101.tar.gz -C $CATALOG_PATH
```

---

## Next Steps

- [User Guide](./USER_GUIDE.md) - Learn how to use Lumina
- [Architecture](./ARCHITECTURE.md) - Understand the system design
- [Development Guide](./CONTRIBUTING.md) - Contribute to the project

---

## Support

- **Issues**: https://github.com/irjudson/lumina/issues
- **Discussions**: https://github.com/irjudson/lumina/discussions
