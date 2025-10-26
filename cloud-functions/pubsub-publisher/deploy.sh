#!/bin/bash

set -e

PROJECT_ID="lexiguard-475609"
REGION="us-central1"
TOPIC_NAME="lexiguard-analysis-jobs"
SUBSCRIPTION_NAME="lexiguard-analysis-jobs-sub"
WORKER_SERVICE="lexiguard-worker"

echo "=========================================="
echo "🚀 Deploying Lexiguard Services"
echo "=========================================="

# 1. Deploy Cloud Function (Publisher)
echo ""
echo "📤 Step 1: Deploying Cloud Function (Firestore Trigger)..."
cd cloud-function

gcloud functions deploy lexiguard-job-publisher \
  --gen2 \
  --runtime=python311 \
  --region=$REGION \
  --source=. \
  --entry-point=publish_analysis_job \
  --trigger-event-filters="type=google.cloud.firestore.document.v1.created" \
  --trigger-event-filters="database=(default)" \
  --trigger-location=$REGION \
  --trigger-event-filters-path-pattern="document=analysisJobs/{docId}" \
  --service-account=372716482731-compute@developer.gserviceaccount.com \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID" \
  --max-instances=10 \
  --memory=256MB \
  --timeout=60s

echo "✅ Cloud Function deployed successfully!"

cd ..

# 2. Verify Pub/Sub Topic exists
echo ""
echo "📋 Step 2: Verifying Pub/Sub Topic..."
if gcloud pubsub topics describe $TOPIC_NAME --project=$PROJECT_ID &> /dev/null; then
    echo "✅ Topic '$TOPIC_NAME' exists"
else
    echo "⚠️  Topic not found. Creating..."
    gcloud pubsub topics create $TOPIC_NAME --project=$PROJECT_ID
    echo "✅ Topic created"
fi

# 3. Get Worker URL
echo ""
echo "🔍 Step 3: Getting Cloud Run Worker URL..."
WORKER_URL=$(gcloud run services describe $WORKER_SERVICE \
  --region=$REGION \
  --project=$PROJECT_ID \
  --format='value(status.url)')

if [ -z "$WORKER_URL" ]; then
    echo "❌ Error: Could not find worker service '$WORKER_SERVICE'"
    echo "   Deploy the worker first with: cd cloud-run-worker && gcloud run deploy"
    exit 1
fi

echo "✅ Worker URL: $WORKER_URL"

# 4. Verify/Update Pub/Sub Subscription
echo ""
echo "📬 Step 4: Verifying Pub/Sub Subscription..."
if gcloud pubsub subscriptions describe $SUBSCRIPTION_NAME --project=$PROJECT_ID &> /dev/null; then
    echo "✅ Subscription exists"
    echo "   Updating push endpoint to: $WORKER_URL"
    
    gcloud pubsub subscriptions update $SUBSCRIPTION_NAME \
      --push-endpoint=$WORKER_URL \
      --project=$PROJECT_ID
    
    echo "✅ Subscription updated"
else
    echo "⚠️  Subscription not found. Creating..."
    
    gcloud pubsub subscriptions create $SUBSCRIPTION_NAME \
      --topic=$TOPIC_NAME \
      --push-endpoint=$WORKER_URL \
      --push-auth-service-account=372716482731-compute@developer.gserviceaccount.com \
      --ack-deadline=600 \
      --message-retention-duration=7d \
      --project=$PROJECT_ID
    
    echo "✅ Subscription created"
fi

# 5. Grant IAM permissions
echo ""
echo "🔐 Step 5: Verifying IAM permissions..."

# Grant Pub/Sub Publisher role to Cloud Function service account
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:372716482731-compute@developer.gserviceaccount.com" \
  --role="roles/pubsub.publisher" \
  --condition=None \
  --quiet

echo "✅ Cloud Function can publish to Pub/Sub"

# Grant Cloud Run Invoker role to Pub/Sub service account
gcloud run services add-iam-policy-binding $WORKER_SERVICE \
  --region=$REGION \
  --member="serviceAccount:372716482731-compute@developer.gserviceaccount.com" \
  --role="roles/run.invoker" \
  --project=$PROJECT_ID \
  --quiet

echo "✅ Pub/Sub can invoke Cloud Run Worker"

# 6. Test deployment
echo ""
echo "=========================================="
echo "🧪 Testing Deployment"
echo "=========================================="

echo ""
echo "Testing Cloud Function logs..."
gcloud functions logs read lexiguard-job-publisher \
  --gen2 \
  --region=$REGION \
  --limit=5 \
  --project=$PROJECT_ID

echo ""
echo "Testing Worker health endpoint..."
curl -s "$WORKER_URL/health" | python3 -m json.tool || echo "⚠️  Health check returned non-JSON response"

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "📊 Status Check Commands:"
echo ""
echo "   # Cloud Function logs:"
echo "   gcloud functions logs read lexiguard-job-publisher --gen2 --region=$REGION --limit=10"
echo ""
echo "   # Worker logs:"
echo "   gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=$WORKER_SERVICE\" --limit=20 --format=\"table(timestamp, textPayload)\""
echo ""
echo "   # Pub/Sub subscription status:"
echo "   gcloud pubsub subscriptions describe $SUBSCRIPTION_NAME"
echo ""
echo "   # Test worker health:"
echo "   curl $WORKER_URL/health"
echo ""
echo "=========================================="
echo "🎯 Next Steps:"
echo "   1. Upload a test document through your app"
echo "   2. Watch the logs with the commands above"
echo "   3. Check Firestore 'analysisJobs' collection for status updates"
echo "=========================================="