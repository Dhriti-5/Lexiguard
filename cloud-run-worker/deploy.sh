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
PUBSUB_TOPIC="document-analysis-jobs"

echo "=========================================="
echo "🚀 Deploying Cloud Run Worker"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo "Service Account: $SERVICE_ACCOUNT"
echo ""

# Get Gemini API Key from environment or prompt
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  GEMINI_API_KEY not found in environment"
    echo ""
    read -p "Enter your Gemini API Key: " GEMINI_API_KEY
    
    if [ -z "$GEMINI_API_KEY" ]; then
        echo "❌ Error: Gemini API Key is required"
        exit 1
    fi
fi

echo ""
echo "📦 Step 1: Deploying Cloud Run service from source..."
echo ""

# Deploy from source (no Dockerfile needed - uses Cloud Build buildpacks)
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
echo "  Memory: 2Gi"
echo "  CPU: 2"
echo "  Timeout: 600s"
echo ""

# Configure Pub/Sub to push to Cloud Run
echo "🔗 Step 2: Configuring Pub/Sub push subscription..."
echo ""

# Check if subscription already exists
if gcloud pubsub subscriptions describe $PUBSUB_SUBSCRIPTION --project=$PROJECT_ID 2>/dev/null; then
    echo "⚠️  Subscription already exists. Deleting and recreating..."
    gcloud pubsub subscriptions delete $PUBSUB_SUBSCRIPTION \
        --project=$PROJECT_ID \
        --quiet
fi

# Create new push subscription
echo "Creating push subscription: $PUBSUB_SUBSCRIPTION"
gcloud pubsub subscriptions create $PUBSUB_SUBSCRIPTION \
    --topic=$PUBSUB_TOPIC \
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
echo "📊 Summary:"
echo "  ✓ Cloud Run service deployed"
echo "  ✓ Environment variables configured"
echo "  ✓ Pub/Sub subscription created"
echo "  ✓ Service account linked"
echo ""
echo "🧪 Testing the deployment:"
echo ""
echo "1. View logs:"
echo "   gcloud run logs read $SERVICE_NAME --region=$REGION --limit=50"
echo ""
echo "2. Monitor Pub/Sub subscription:"
echo "   gcloud pubsub subscriptions describe $PUBSUB_SUBSCRIPTION"
echo ""
echo "3. Test end-to-end:"
echo "   - Upload a document through your frontend"
echo "   - Check job status in Firestore"
echo "   - Monitor worker logs for processing"
echo ""
echo "📝 Important Notes:"
echo "  • Service is NOT publicly accessible (--no-allow-unauthenticated)"
echo "  • Only Pub/Sub can trigger the service via service account"
echo "  • Logs are available in Cloud Logging"
echo "  • Auto-scaling: 0 to 10 instances based on load"
echo ""
echo "🎉 Happy Processing!"
echo ""