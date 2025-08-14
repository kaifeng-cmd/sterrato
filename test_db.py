import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

try:
    collection_available = client.get_collections()
    collection_info = client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
    print(f"Available collections in Qdrant: {collection_available}")
    print(f"Succeed to access Qdrant Cloud collection：{collection_info}")
except Exception as e:
    print(f"Failed to access：{e}")
