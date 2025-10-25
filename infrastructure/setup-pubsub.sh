#!/bin/bash

# ========================================
# LexiGuard Infrastructure Setup Script
# ========================================
# This script sets up all required Google Cloud resources
# Run this ONCE before deploying Cloud Functions and Cloud Run

set -e  # Exit on any error

# ========================================
# Configuration Variables
# ========================================
PROJECT_ID="lexiguard-475609"  # REPLACE WITH YOUR PROJECT ID
REGION="us-central1"           # Change if needed
BUCKET_NAME="lexiguard-documents"
TOPIC_NAME="document-analysis-jobs"
SUBSCRIPTION_NAME="document-analysis-worker"
SERVICE_ACCOUNT_NAME="lexiguard-worker"

echo "🚀 Starting LexiGuard Infrastructure Setup..."
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"

# ========================================
# 1. Set active project
# ========================================
echo ""
echo "📌 Step 1: Setting active project..."
gcloud config set project $PROJECT_ID

# ========================================
# 2. Enable required APIs
# ========================================
echo ""
echo "📌 Step 2: Enabling required Google Cloud APIs..."
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable pubsub.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable dlp.googleapis.com
gcloud services enable firestore.googleapis.com
gcloud services enable cloudbuild.googleapis.com

echo "✅ APIs enabled successfully"

# ========================================
# 3. Create Cloud Storage bucket
# ========================================
echo ""
echo "📌 Step 3: Creating Cloud Storage bucket..."
if gsutil ls -b gs://$BUCKET_NAME 2>/dev/null; then
    echo "⚠️  Bucket already exists: gs://$BUCKET_NAME"
else
    gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME
    echo "✅ Bucket created: gs://$BUCKET_NAME"
fi

# Create folder structure
gsutil mkdir gs://$BUCKET_NAME/uploads/ || true
gsutil mkdir gs://$BUCKET_NAME/processed/ || true

# ========================================
# 4. Create Pub/Sub Topic
# ========================================
echo ""
echo "📌 Step 4: Creating Pub/Sub topic..."
if gcloud pubsub topics describe $TOPIC_NAME 2>/dev/null; then
    echo "⚠️  Topic already exists: $TOPIC_NAME"
else
    gcloud pubsub topics create $TOPIC_NAME
    echo "✅ Topic created: $TOPIC_NAME"
fi

# ========================================
# 5. Create Pub/Sub Subscription
# ========================================
echo ""
echo "📌 Step 5: Creating Pub/Sub subscription..."
if gcloud pubsub subscriptions describe $SUBSCRIPTION_NAME 2>/dev/null; then
    echo "⚠️  Subscription already exists: $SUBSCRIPTION_NAME"
else
    gcloud pubsub subscriptions create $SUBSCRIPTION_NAME \
        --topic=$TOPIC_NAME \
        --ack-deadline=600 \
        --message-retention-duration=7d
    echo "✅ Subscription created: $SUBSCRIPTION_NAME"
fi

# ========================================
# 6. Create Service Account
# ========================================
echo ""
echo "📌 Step 6: Creating service account for Cloud Run worker..."
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if gcloud iam service-accounts describe $SERVICE_ACCOUNT_EMAIL 2>/dev/null; then
    echo "⚠️  Service account already exists: $SERVICE_ACCOUNT_EMAIL"
else
    gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
        --display-name="LexiGuard Worker Service Account"
    echo "✅ Service account created: $SERVICE_ACCOUNT_EMAIL"
fi

# ========================================
# 7. Grant IAM Permissions
# ========================================
echo ""
echo "📌 Step 7: Granting IAM permissions..."

# Permissions for Cloud Run worker
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/pubsub.subscriber" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/storage.objectAdmin" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/datastore.user" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/dlp.user" \
    --quiet

echo "✅ IAM permissions granted"

# ========================================
# 8. Create service account key (for local testing)
# ========================================
echo ""
echo "📌 Step 8: Creating service account key for local development..."
KEY_FILE="./lexiguard-service-account-key.json"

if [ -f "$KEY_FILE" ]; then
    echo "⚠️  Key file already exists: $KEY_FILE"
else
    gcloud iam service-accounts keys create $KEY_FILE \
        --iam-account=$SERVICE_ACCOUNT_EMAIL
    echo "✅ Service account key created: $KEY_FILE"
    echo "⚠️  IMPORTANT: Keep this key secure and never commit it to Git!"
fi

# ========================================
# Summary
# ========================================
echo ""
echo "=========================================="
echo "✅ Infrastructure Setup Complete!"
echo "=========================================="
echo ""
echo "📋 Resources Created:"
echo "  • Cloud Storage Bucket: gs://$BUCKET_NAME"
echo "  • Pub/Sub Topic: $TOPIC_NAME"
echo "  • Pub/Sub Subscription: $SUBSCRIPTION_NAME"
echo "  • Service Account: $SERVICE_ACCOUNT_EMAIL"
echo ""
echo "📝 Next Steps:"
echo "  1. Update shared/constants.py with your PROJECT_ID and BUCKET_NAME"
echo "  2. Deploy Cloud Function: cd cloud-functions/pubsub-publisher && ./deploy.sh"
echo "  3. Deploy Cloud Run Worker: cd cloud-run-worker && ./deploy.sh"
echo "  4. Update backend .env file with new environment variables"
echo ""
echo "🔑 Service Account Key: $KEY_FILE"
echo "   Set this in GOOGLE_APPLICATION_CREDENTIALS environment variable"
echo ""