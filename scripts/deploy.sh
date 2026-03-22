#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  deploy.sh — Build and deploy Crop Disease Classifier
#  Usage:
#    ./scripts/deploy.sh              # build + run locally
#    ./scripts/deploy.sh --prod       # with nginx
#    ./scripts/deploy.sh --push TAG   # push to Docker Hub
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

IMAGE="crop-disease-classifier"
TAG="${TAG:-latest}"
PUSH=false
PROD=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --push)  PUSH=true; TAG="${2:-latest}"; shift 2 ;;
    --prod)  PROD=true; shift ;;
    --tag)   TAG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "======================================"
echo "  Crop Disease Classifier — Deploy"
echo "  Image: ${IMAGE}:${TAG}"
echo "======================================"

# ── Check model exists ────────────────────────────────────────────────────────
if [ ! -f "models/best_model.pth" ]; then
  echo ""
  echo "⚠️  WARNING: models/best_model.pth not found."
  echo "   The app will start in demo mode."
  echo "   Train a model first:"
  echo "     python data/download_dataset.py --source sample"
  echo "     python src/train.py --data ./data/plantvillage"
  echo ""
fi

# ── Build ─────────────────────────────────────────────────────────────────────
echo "Building Docker image..."
docker build -t "${IMAGE}:${TAG}" .
echo "✓ Build complete"

# ── Push ─────────────────────────────────────────────────────────────────────
if [ "$PUSH" = true ]; then
  REGISTRY="${DOCKER_REGISTRY:-}"
  if [ -n "$REGISTRY" ]; then
    docker tag "${IMAGE}:${TAG}" "${REGISTRY}/${IMAGE}:${TAG}"
    docker push "${REGISTRY}/${IMAGE}:${TAG}"
    echo "✓ Pushed to ${REGISTRY}/${IMAGE}:${TAG}"
  else
    echo "Set DOCKER_REGISTRY env var to push (e.g. export DOCKER_REGISTRY=yourdockerhubuser)"
    exit 1
  fi
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Starting services..."
if [ "$PROD" = true ]; then
  docker compose --profile production up -d
  echo "✓ Running with Nginx on http://localhost:80"
else
  docker compose up -d crop-classifier
  echo "✓ Running on http://localhost:8000"
fi

echo ""
echo "Useful commands:"
echo "  docker compose logs -f crop-classifier   # stream logs"
echo "  docker compose down                      # stop"
echo "  curl http://localhost:8000/health        # health check"
