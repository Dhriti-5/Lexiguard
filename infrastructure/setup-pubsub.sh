#!/bin/bash

# Setup Pub/Sub infrastructure for LexiGuard async processing
# Creates topic and subscription with proper IAM permissions

set -e

echo "🚀 Setting up LexiGuard Pub/Sub Infrastructure..."
echo ""

# Configuration
PROJECT_ID="lexiguard-475609"
SERVICE_ACCOUNT="372716482731-compute@developer.gserviceaccount.com"
TOPIC_NAME="lexiguard-analysis-jobs"
SUBSCRIPTION_NAME="lexiguard-analysis-jobs-sub"

echo "📋 Configuration:"
echo "   Project: $PROJECT_ID"
echo "   Topic: $TOPIC_NAME"
echo "   Subscription: $SUBSCRIPTION_NAME"
echo "   Service Account: $SERVICE_ACCOUNT"
echo ""

# Set active project
gcloud config set project $PROJECT_ID

# Step 1: Enable required APIs
echo "📦 Enabling required APIs..."
gcloud services enable \
  pubsub.googleapis.com \
  run.googleapis.com \
  cloudfunctions.googleapis.com \
  firestore.googleapis.com \
  storage.googleapis.com \
  dlp.googleapis.com

echo "✅ APIs enabled"
echo ""

# Step 2: Create Pub/Sub topic (if not exists)
echo "📝 Creating Pub/Sub topic..."
if gcloud pubsub topics describe $TOPIC_NAME &>/dev/null; then
    echo "   Topic already exists: $TOPIC_NAME"
else
    gcloud pubsub topics create $TOPIC_NAME
    echo "   ✅ Topic created: $TOPIC_NAME"
fi
echo ""

# Step 3: Grant IAM permissions to service account
echo "🔐 Setting up IAM permissions..."

# Publisher role (for Cloud Function)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/pubsub.publisher" \
  --condition=None \
  --quiet

echo "   ✅ Granted Pub/Sub Publisher role"

# Subscriber role (for Cloud Run)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/pubsub.subscriber" \
  --condition=None \
  --quiet

echo "   ✅ Granted Pub/Sub Subscriber role"

# Storage Object Viewer (to read uploaded files)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/storage.objectViewer" \
  --condition=None \
  --quiet

echo "   ✅ Granted Storage Object Viewer role"

# Firestore User (to read/write Firestore)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/datastore.user" \
  --condition=None \
  --quiet

echo "   ✅ Granted Datastore User role"

# DLP User (for PII redaction)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/dlp.user" \
  --condition=None \
  --quiet

echo "   ✅ Granted DLP User role"
echo ""

# Step 4: Check if Cloud Run service exists
echo "🔍 Checking if Cloud Run worker is deployed..."
REGION="us-central1"
SERVICE_NAME="lexiguard-worker"

if gcloud run services describe $SERVICE_NAME --region=$REGION &>/dev/null; then
    echo "   ✅ Cloud Run service found"
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
      --region=$REGION \
      --format='value(status.url)')
    
    echo "   Service URL: $SERVICE_URL"
    echo ""
    
    # Step 5: Create or update Pub/Sub subscription
    echo "📮 Creating Pub/Sub push subscription..."
    
    if gcloud pubsub subscriptions describe $SUBSCRIPTION_NAME &>/dev/null; then
        echo "   ⚠️  Subscription already exists, deleting old one..."
        gcloud pubsub subscriptions delete $SUBSCRIPTION_NAME --quiet
    fi
    
    gcloud pubsub subscriptions create $SUBSCRIPTION_NAME \
      --topic=$TOPIC_NAME \
      --push-endpoint="$SERVICE_URL" \
      --push-auth-service-account="$SERVICE_ACCOUNT" \
      --ack-deadline=600 \
      --message-retention-duration=7d \
      --quiet
    
    echo "   ✅ Subscription created: $SUBSCRIPTION_NAME"
    echo ""
    
else
    echo "   ⚠️  Cloud Run service not deployed yet"
    echo ""
    echo "   Deploy the worker first using:"
    echo "   cd cloud-run-worker && bash deploy.sh"
    echo ""
    echo "   Then run this script again to create the subscription"
    echo ""
    exit 0
fi

# Step 6: Verify setup
echo "✅ Infrastructure setup complete!"
echo ""
echo "📊 Verification:"
echo ""
echo "1. Topic created:"
echo "   gcloud pubsub topics list | grep $TOPIC_NAME"
echo ""
echo "2. Subscription created:"
echo "   gcloud pubsub subscriptions list | grep $SUBSCRIPTION_NAME"
echo ""
echo "3. Test the setup by uploading a document via the frontend"
echo ""
echo "4. Monitor logs:"
echo "   Cloud Function: gcloud functions logs read lexiguard-job-publisher --gen2 --limit=20"
echo "   Cloud Run: gcloud run logs read $SERVICE_NAME --limit=20"
echo ""