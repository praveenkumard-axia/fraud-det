#!/bin/bash
set -e

# Registry User
REGISTRY_USER="pduraiswamy16722"

echo "🐳 Logging into Docker Hub... (if needed)"
# docker login -u $REGISTRY_USER

echo "🚀 Building and Pushing Images..."

# 1. Backend Server
echo "📦 Building Backend..."
docker build -t $REGISTRY_USER/fraud-backend:latest -f Dockerfile.backend .
docker push $REGISTRY_USER/fraud-backend:latest

# 2. Data Gather
echo "📦 Building Data Gather..."
docker build -t $REGISTRY_USER/fraud-data-gather:latest ./pods/data-gather
docker push $REGISTRY_USER/fraud-data-gather:latest

# 3. Data Prep
echo "📦 Building Data Prep..."
docker build -t $REGISTRY_USER/fraud-data-prep:latest ./pods/data-prep
docker push $REGISTRY_USER/fraud-data-prep:latest

# 4. Model Build
echo "📦 Building Model Build..."
docker build -t $REGISTRY_USER/fraud-model-build:latest ./pods/model-build
docker push $REGISTRY_USER/fraud-model-build:latest

# 5. Inference
echo "📦 Building Inference..."
docker build -t $REGISTRY_USER/fraud-inference:latest ./pods/inference
docker push $REGISTRY_USER/fraud-inference:latest

echo "✅ All images built and pushed successfully!"
