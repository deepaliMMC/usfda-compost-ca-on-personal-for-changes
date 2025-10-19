#!/bin/bash
# Sync training data from S3 to local storage

set -e

source .env

echo "Syncing training data from S3..."

# Sync images
echo "Downloading training images..."
aws s3 sync \
    s3://${S3_BUCKET}/${S3_IMG_PREFIX} \
    ${LOCAL_IMG_DIR} \
    --region ${AWS_REGION} \
    --request-payer requester

# Sync masks
echo "Downloading training masks..."
aws s3 sync \
    s3://${S3_BUCKET}/${S3_MSK_PREFIX} \
    ${LOCAL_MSK_DIR} \
    --region ${AWS_REGION} \
    --request-payer requester

echo "Sync complete!"
echo "Images: ${LOCAL_IMG_DIR}"
echo "Masks: ${LOCAL_MSK_DIR}"
