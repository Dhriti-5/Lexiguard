#!/bin/bash

# Deploy Cloud Run Worker for document processing
# Uses source-based deployment (NO Docker required)
# Cloud Run will automatically build from source code

set -e

echo "🚀 Deploying LexiGuard Cloud Run Worker (Source-based)..."

# Configuration
PROJECT_ID="lexiguard-475609"
REGION="us-central1"
SERVICE_NAME="lexiguard-worker"
SERVICE_ACCOUNT="372716482731-compute@developer.gserviceaccount.com"

# ⚠️ IMPORTANT: Set your Gemini API key here
# Get it from: https://makersuite.google.com/app/apikey
read -p "Enter your Gemini API Key: " GEMINI_API_KEY

if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ ERROR: Gemini API key is required"
    echo "   Get your API key from: https://makersuite.google.com/app/apikey"
    exit 1
fi

echo ""
echo "📋 Configuration:"
echo "   Project: $PROJECT_ID"
echo "   Region: $REGION"
echo "   Service: $SERVICE_NAME"
echo "   Service Account: $SERVICE_ACCOUNT"
echo "   Bucket: lexiguard-documents"
echo ""

# Verify we're in the correct directory
if [ ! -f "main.py" ] || [ ! -f "requirements.txt" ]; then
    echo "❌ ERROR: main.py or requirements.txt not found"
    echo "   Make sure you're in the cloud-run-worker directory"
    exit 1
fi

echo "✅ Found main.py and requirements.txt"
echo ""

# Deploy to Cloud Run using source deployment
echo "📦 Deploying to Cloud Run (this may take 2-3 minutes)..."
echo ""

gcloud run deploy $SERVICE_NAME \
  --source=. \
  --platform=managed \
  --region=$REGION \
  --project=$PROJECT_ID \
  --service-account="$SERVICE_ACCOUNT" \
  --set-env-vars="GCP_PROJECT=$PROJECT_ID,GCS_BUCKET_NAME=lexiguard-documents,GOOGLE_API_KEY=$GEMINI_API_KEY" \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --max-instances=10 \
  --min-instances=0 \
  --allow-unauthenticated \
  --quiet

echo ""
echo "✅ Cloud Run Worker deployed successfully!"
echo ""

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --region=$REGION \
  --project=$PROJECT_ID \
  --format='value(status.url)')

echo "🌐 Worker URL: $SERVICE_URL"
echo ""

# Test health endpoint
echo "🏥 Testing health endpoint..."
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL/health")

if [ "$HEALTH_STATUS" = "200" ]; then
    echo "✅ Worker is healthy!"
else
    echo "⚠️  Health check returned: $HEALTH_STATUS"
fi
echo ""

echo "⚠️ NEXT STEP: Create Pub/Sub subscription"
echo ""
echo "Run this command to create the subscription:"
echo ""
echo "gcloud pubsub subscriptions create lexiguard-analysis-jobs-sub \\"
echo "  --topic=lexiguard-analysis-jobs \\"
echo "  --push-endpoint=\"$SERVICE_URL\" \\"
echo "  --push-auth-service-account=\"$SERVICE_ACCOUNT\" \\"
echo "  --ack-deadline=600 \\"
echo "  --message-retention-duration=7d"
echo ""
echo "Or run: bash ../infrastructure/setup-pubsub.sh"
echo ""
echo "🔍 Useful commands:"
echo ""
echo "View logs:"
echo "  gcloud run logs read $SERVICE_NAME --region=$REGION --limit=50"
echo ""
echo "View service details:"
echo "  gcloud run services describe $SERVICE_NAME --region=$REGION"
echo ""
echo "Update environment variables:"
echo "  gcloud run services update $SERVICE_NAME --region=$REGION --update-env-vars=KEY=VALUE"
echo ""