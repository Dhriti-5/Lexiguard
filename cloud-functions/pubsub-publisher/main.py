import json
import logging
import os
import base64
from google.cloud import pubsub_v1

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pub/Sub publisher
publisher = pubsub_v1.PublisherClient()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "lexiguard-475609")
TOPIC_NAME = "lexiguard-analysis-jobs"
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

logger.info(f"📋 Cloud Function initialized")
logger.info(f"   Project: {PROJECT_ID}")
logger.info(f"   Topic: {topic_path}")

def publish_analysis_job(event, context):
    """
    Triggered by Firestore onCreate event on analysisJobs collection.
    Publishes job details to Pub/Sub for worker processing.
    
    Args:
        event: The Firestore event (CloudEvent format - bytes or dict)
        context: The Cloud Functions context
    """
    try:
        logger.info("=" * 80)
        logger.info("🔔 Firestore trigger activated")
        logger.info(f"   Context resource: {context.resource if context else 'None'}")

        # ✅ STEP 1: Decode event safely
        event_dict = None
        if isinstance(event, bytes):
            try:
                # Attempt base64 decode (CloudEvent Gen2 standard)
                decoded = base64.b64decode(event).decode("utf-8")
                event_dict = json.loads(decoded)
                logger.info("   Successfully decoded base64 Firestore event")
            except Exception as e:
                logger.warning(f"⚠️ Base64 decode failed, trying UTF-8 decode fallback: {e}")
                try:
                    event_dict = json.loads(event.decode("utf-8"))
                    logger.info("   Successfully decoded UTF-8 bytes event")
                except Exception as e2:
                    logger.error(f"❌ Unable to decode Firestore event: {e2}")
                    return
        elif isinstance(event, dict):
            event_dict = event
            logger.info("   Event is already dict")
        else:
            logger.error(f"❌ Unexpected event type: {type(event)}")
            return

        logger.info(f"   Event keys: {list(event_dict.keys()) if isinstance(event_dict, dict) else 'N/A'}")

        # ✅ STEP 2: Parse Firestore document fields
        fields = {}
        try:
            if "data" in event_dict:
                data_section = event_dict["data"]
                if isinstance(data_section, str):
                    data_section = json.loads(data_section)
                value_section = data_section.get("value", {})
                fields = value_section.get("fields", {})
                logger.info(f"   Found {len(fields)} fields in document")
            elif "value" in event_dict and "fields" in event_dict["value"]:
                fields = event_dict["value"]["fields"]
                logger.info("   Using legacy Firestore event format")
            else:
                logger.error(f"❌ Cannot find Firestore fields in event structure")
                logger.error(f"   Available keys: {event_dict.keys()}")
                return
        except Exception as parse_error:
            logger.error(f"❌ Failed to parse event structure: {parse_error}")
            logger.error(f"   Event dict (truncated): {json.dumps(event_dict, indent=2)[:500]} ...")
            return

        # ✅ STEP 3: Extract typed Firestore fields
        def get_field_value(field_name, default=""):
            field = fields.get(field_name, {})
            if not isinstance(field, dict):
                return str(field) if field else default
            if "stringValue" in field:
                return field["stringValue"]
            if "integerValue" in field:
                return int(field["integerValue"])
            if "booleanValue" in field:
                return field["booleanValue"]
            if "timestampValue" in field:
                return field["timestampValue"]
            return default

        job_message = {
            "jobId": get_field_value("jobId"),
            "userID": get_field_value("userID"),
            "documentTitle": get_field_value("documentTitle"),
            "originalFilename": get_field_value("originalFilename"),
            "fileType": get_field_value("fileType"),
            "gcsPath": get_field_value("gcsPath"),
            "analysisType": get_field_value("analysisType", "standard"),
            "status": get_field_value("status"),
        }

        logger.info(f"📦 Extracted job details: {json.dumps(job_message, indent=2)}")

        # ✅ STEP 4: Skip non-pending jobs
        if job_message["status"] != "pending":
            logger.info(f"⏭️ Skipping - status is '{job_message['status']}'")
            return

        # ✅ STEP 5: Validate required fields
        required = ["jobId", "userID", "gcsPath", "fileType"]
        missing = [f for f in required if not job_message.get(f)]
        if missing:
            logger.error(f"❌ Missing required fields: {missing}")
            return

        # ✅ STEP 6: Publish to Pub/Sub
        message_bytes = json.dumps(job_message).encode("utf-8")
        logger.info("📤 Publishing to Pub/Sub...")

        future = publisher.publish(topic_path, message_bytes)
        message_id = future.result(timeout=30)

        logger.info(f"✅ Successfully published to Pub/Sub!")
        logger.info(f"   Message ID: {message_id}")
        logger.info(f"   Job ID: {job_message['jobId']}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ Error in publish_analysis_job: {e}")
        logger.error(f"   Type: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("=" * 80)
        return
