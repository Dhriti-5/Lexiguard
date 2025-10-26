import functions_framework
from google.cloud import pubsub_v1
from google.events.cloud import firestore
from cloudevents.http import CloudEvent
import os
import json

# Initialize Pub/Sub publisher
publisher = pubsub_v1.PublisherClient()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "lexiguard-475609")
TOPIC_NAME = "lexiguard-analysis-jobs"
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

print(f"📋 Cloud Function initialized")
print(f"   Project: {PROJECT_ID}")
print(f"   Topic: {topic_path}")

@functions_framework.cloud_event
def publish_analysis_job(cloud_event: CloudEvent):
    """
    Triggered by Firestore onCreate event on analysisJobs collection.
    Publishes job details to Pub/Sub for worker processing.
    """
    try:
        print("=" * 80)
        print("🔔 Firestore trigger activated")
        print(f"   Event type: {cloud_event['type']}")
        print(f"   Event source: {cloud_event['source']}")
        
        # Parse Firestore document data from Protobuf
        firestore_payload = firestore.DocumentEventData()
        firestore_payload._pb.ParseFromString(cloud_event.data)
        
        # Get the document that was created
        document = firestore_payload.value
        fields = document.fields
        
        # Extract document ID from the event subject
        subject = cloud_event.get("subject", "")
        doc_id = subject.split("/")[-1] if subject else None
        
        print(f"   Document ID: {doc_id}")
        print(f"   Fields count: {len(fields)}")
        
        # Helper function to extract Firestore field values
        def get_field_value(field_name, default=""):
            if field_name not in fields:
                return default
            
            field = fields[field_name]
            
            # The field object has different value types as attributes
            # We check which one is set (non-empty)
            if field.string_value:
                return field.string_value
            elif field.integer_value:
                return int(field.integer_value)
            elif field.double_value:
                return field.double_value
            # boolean_value can be False, so check explicitly
            elif hasattr(field, 'boolean_value'):
                return field.boolean_value
            elif field.timestamp_value:
                # Convert timestamp to ISO string
                ts = field.timestamp_value
                return ts.ToJsonString() if hasattr(ts, 'ToJsonString') else str(ts)
            else:
                return default
        
        # Extract job data
        job_message = {
            "jobId": get_field_value("jobId") or doc_id,
            "userID": get_field_value("userID"),
            "documentTitle": get_field_value("documentTitle"),
            "originalFilename": get_field_value("originalFilename"),
            "fileType": get_field_value("fileType"),
            "gcsPath": get_field_value("gcsPath"),
            "analysisType": get_field_value("analysisType", "standard"),
            "status": get_field_value("status"),
        }
        
        print(f"📦 Extracted job details:")
        print(json.dumps(job_message, indent=2))
        
        # Only process pending jobs
        if job_message["status"] != "pending":
            print(f"⏭️ Skipping - status is '{job_message['status']}'")
            return
        
        # Validate required fields
        required = ["jobId", "userID", "gcsPath", "fileType"]
        missing = [f for f in required if not job_message.get(f)]
        if missing:
            print(f"❌ Missing required fields: {missing}")
            return
        
        # Publish to Pub/Sub
        message_bytes = json.dumps(job_message).encode("utf-8")
        print("📤 Publishing to Pub/Sub...")
        
        future = publisher.publish(topic_path, message_bytes)
        message_id = future.result(timeout=30)
        
        print(f"✅ Successfully published to Pub/Sub!")
        print(f"   Message ID: {message_id}")
        print(f"   Job ID: {job_message['jobId']}")
        print("=" * 80)
        
    except Exception as e:
        print("=" * 80)
        print(f"❌ Error in publish_analysis_job: {e}")
        print(f"   Type: {type(e).__name__}")
        import traceback
        print(traceback.format_exc())
        print("=" * 80)
        return