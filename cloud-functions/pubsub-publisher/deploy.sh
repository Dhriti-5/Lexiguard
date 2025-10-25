#!/bin/bash

# ========================================
# Deploy Cloud Run Worker (Source-based, No Docker)
# ========================================
# This deploys directly from source code using Cloud Build

set -e

# Configuration
PROJECT_ID="lexiguard-475609"
REGION="us-central1"
SERVICE_NAME="lexiguard-worker"
SERVICE_ACCOUNT="372716482731-compute@developer.gserviceaccount.com"  # ✅ YOUR SERVICE ACCOUNT
BUCKET_NAME="lexiguard-documents"
PUBSUB_SUBSCRIPTION="document-analysis-worker"

# Get Gemini API Key from environment or prompt
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  GEMINI_API_KEY not found in environment"
    read -p "Enter your Gemini API Key: " GEMINI_API_KEY
fi

echo "🚀 Deploying Cloud Run Worker: $SERVICE_NAME"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service Account: $SERVICE_ACCOUNT"

# Deploy from source (no Dockerfile needed)
gcloud run deploy $SERVICE_NAME \
    --source=. \
    --platform=managed \
    --region=$REGION \
    --project=$PROJECT_ID \
    --service-account=$SERVICE_ACCOUNT \
    --set-env-vars="GCP_PROJECT=$PROJECT_ID,GCS_BUCKET_NAME=$BUCKET_NAME,GEMINI_API_KEY=$GEMINI_API_KEY" \
    --memory=2Gi \
    --cpu=2 \
    --timeout=600 \
    --concurrency=10 \
    --min-instances=0 \
    --max-instances=10 \
    --no-allow-unauthenticated

echo ""
echo "✅ Cloud Run service deployed successfully!"

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --region=$REGION \
    --project=$PROJECT_ID \
    --format="value(status.url)")

echo ""
echo "📋 Service Details:"
echo "  Service Name: $SERVICE_NAME"
echo "  Service URL: $SERVICE_URL"
echo "  Region: $REGION"
echo "  Service Account: $SERVICE_ACCOUNT"

# Configure Pub/Sub to push to Cloud Run
echo ""
echo "🔗 Configuring Pub/Sub push subscription..."

# Delete existing subscription if it exists
gcloud pubsub subscriptions delete $PUBSUB_SUBSCRIPTION \
    --project=$PROJECT_ID \
    --quiet 2>/dev/null || true

# Create new push subscription
gcloud pubsub subscriptions create $PUBSUB_SUBSCRIPTION \
    --topic=document-analysis-jobs \
    --push-endpoint=$SERVICE_URL \
    --push-auth-service-account=$SERVICE_ACCOUNT \
    --ack-deadline=600 \
    --message-retention-duration=7d \
    --project=$PROJECT_ID

echo ""
echo "✅ Pub/Sub push subscription configured!"

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "🧪 Test the deployment:"
echo "  1. Upload a document through your frontend"
echo "  2. Check logs: gcloud run logs read $SERVICE_NAME --region=$REGION --project=$PROJECT_ID"
echo "  3. Monitor Firestore for job status updates"
echo "  4. View service: gcloud run services describe $SERVICE_NAME --region=$REGION --project=$PROJECT_ID"
echo ""