#!/usr/bin/env bash
# Push backend and frontend Docker images to Amazon ECR.
#
# Usage:
#   export AWS_ACCOUNT_ID=123456789012
#   export AWS_REGION=us-west-2
#   bash deploy/push-to-ecr.sh

set -euo pipefail

ACCOUNT=${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}
REGION=${AWS_REGION:-us-west-2}
ECR_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

echo "── Authenticating with ECR ──"
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ECR_URI}"

echo "── Creating repositories (if needed) ──"
for repo in clinical-pal-backend clinical-pal-frontend; do
  aws ecr describe-repositories --repository-names "${repo}" --region "${REGION}" 2>/dev/null || \
    aws ecr create-repository --repository-name "${repo}" --region "${REGION}"
done

echo "── Building and pushing backend ──"
docker build -t clinical-pal-backend -f Dockerfile.backend .
docker tag clinical-pal-backend:latest "${ECR_URI}/clinical-pal-backend:latest"
docker push "${ECR_URI}/clinical-pal-backend:latest"

echo "── Building and pushing frontend ──"
docker build -t clinical-pal-frontend -f Dockerfile.frontend .
docker tag clinical-pal-frontend:latest "${ECR_URI}/clinical-pal-frontend:latest"
docker push "${ECR_URI}/clinical-pal-frontend:latest"

echo "── Done! Images pushed to ECR ──"
echo "Backend: ${ECR_URI}/clinical-pal-backend:latest"
echo "Frontend: ${ECR_URI}/clinical-pal-frontend:latest"
