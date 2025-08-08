import os
import firebase_admin
from firebase_admin import credentials, firestore
import logging

logger = logging.getLogger(__name__)

db = None
FIREBASE_AVAILABLE = False

def initialize_firestore():
    global db, FIREBASE_AVAILABLE
    
    if firebase_admin._apps:
        db = firestore.client()
        FIREBASE_AVAILABLE = True
        logger.info("Firebase app already initialized.")
        return db, FIREBASE_AVAILABLE

    # Default to service-account-key.json in the root directory
    cred_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'service-account-key.json')

    if os.path.exists(cred_path):
        try:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            FIREBASE_AVAILABLE = True
            logger.info(f"Firebase initialized successfully using key at: {cred_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase with service account key at {cred_path}: {e}")
            db = None
            FIREBASE_AVAILABLE = False
    else:
        logger.warning(f"Service account key not found at the expected path: {cred_path}. Firestore is not available.")
        db = None
        FIREBASE_AVAILABLE = False
        
    return db, FIREBASE_AVAILABLE

# Initialize on import
db, FIREBASE_AVAILABLE = initialize_firestore()
