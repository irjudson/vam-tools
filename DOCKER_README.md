# VAM Tools - Docker Quick Start

Run VAM Tools with Docker for easy deployment with GPU-accelerated background processing.

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
