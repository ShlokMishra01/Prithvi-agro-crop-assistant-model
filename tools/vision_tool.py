import os
import mimetypes
import requests
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.tools import tool
from .tool_schemas import VisionInput

load_dotenv()

API_URL = os.getenv(
    "VISION_API_URL",
    "https://tweizy-crop-disease-classifier-api.hf.space/classify/",
)


def _vision_timeout_seconds() -> int:
    try:
        return max(5, int(os.getenv("VISION_API_TIMEOUT_SECONDS", "30")))
    except ValueError:
        return 30


def _max_image_bytes() -> int:
    try:
        return max(1, int(os.getenv("AGRIBOT_MAX_IMAGE_MB", "8"))) * 1024 * 1024
    except ValueError:
        return 8 * 1024 * 1024

@tool
def vision_classify(image_path: str) -> Dict[str, Any]:
    """Classify crop disease from a local image path using the deployed vision API."""
    try:
        _ = VisionInput(image_path=image_path)
    except Exception as e:
        return {"error": f"Invalid vision input: {e}"}
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}
    if os.path.getsize(image_path) > _max_image_bytes():
        return {"error": "Image is larger than the configured upload limit."}
    try:
        with open(image_path, "rb") as f:
            mime_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
            files = {"file": (os.path.basename(image_path), f, mime_type)}
            resp = requests.post(API_URL, files=files, timeout=_vision_timeout_seconds())
            resp.raise_for_status()
            response_data = resp.json()
            if not isinstance(response_data, dict):
                return {"error": "Vision API returned an unexpected response format."}
            return response_data
    except Exception as e:
        return {"error": f"Vision API request failed: {e}"}
