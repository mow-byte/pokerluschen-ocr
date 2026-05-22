from google import genai
from google.genai.types import Part
from fastapi import FastAPI, UploadFile, File, Query
import json
from typing import List
from difflib import get_close_matches
import os
import io
from PIL import Image
import logging
import time

app = FastAPI()

# simple stdout logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# --- CONFIG ---
PROJECT_ID = "temporal-ground-492911-k4"
LOCATION = "us-central1"  # or us-central1

# --- INIT Google GenAI client (uses service account automatically) ---
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

MODEL_NAME = "gemini-2.5-flash"

def extract_json(text):
    return json.loads(text)

def fix_name(name, valid_names):
    match = get_close_matches(name, valid_names, n=1, cutoff=0.6)
    return match[0] if match else None
# --- ENDPOINT ---
@app.post("/extract")
async def extract(file: UploadFile = File(...), names: List[str] = Query(...), max_dim: int = Query(1600), quality: int = Query(85)):
    image_bytes = await file.read()
    logger.info("Received upload: filename=%s content_type=%s size=%d bytes", getattr(file, 'filename', None), file.content_type, len(image_bytes))

    def resize_image_bytes(image_bytes: bytes, max_dim: int = 1600, quality: int = 85):
        start = time.time()
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                orig_mode = img.mode
                w, h = img.size
                logger.info("Original image: mode=%s size=%dx%d", orig_mode, w, h)

                # ensure RGB (no alpha) for JPEG
                if img.mode in ("RGBA", "LA"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                w, h = img.size
                scale = min(1.0, float(max_dim) / max(w, h))
                if scale < 1.0:
                    new_size = (int(w * scale), int(h * scale))
                    img = img.resize(new_size, Image.LANCZOS)
                    logger.info("Resized image to %dx%d (max_dim=%d)", new_size[0], new_size[1], max_dim)
                else:
                    logger.info("No resizing needed (max_dim=%d)", max_dim)

                out = io.BytesIO()
                img.save(out, format="JPEG", quality=quality, optimize=True)
                resized = out.getvalue()
                logger.info("Image compressed: %d bytes (quality=%d) in %.3fs", len(resized), quality, time.time() - start)
                return resized, "image/jpeg"
        except Exception as e:
            logger.exception("Image resize/compress failed, using original bytes: %s", e)
            return image_bytes, file.content_type

    resized_bytes, resized_mime = resize_image_bytes(image_bytes, max_dim=max_dim, quality=quality)
    logger.info("Using mime=%s size=%d bytes for model", resized_mime, len(resized_bytes))

    image_part = Part.from_bytes(data=resized_bytes, mime_type=resized_mime)

    # --- PROMPT ---
    prompt = f"""
    You are extracting structured data from a handwritten scorecard.

    Players must ONLY be from this list:
    {names}

    Rules:
    - Do NOT invent names
    - If unclear, map to closest valid name
    - Count how many times "5" appears in each player's row (buy-ins) in the second column from the left
    - Extract the payout € value in the third column from the left
    - Extract the result € value from the far right column (can be positive or negative)
    - Each player appears at most once
    - Return ONLY raw JSON.
    - Do NOT use markdown.
    - Do NOT wrap in ``` or ```json.
    - If the name contains 'Gast', use the name 'Gast' and set guest to true.
    - All numbers must be numeric, not strings.
    [
    {{"name": "Sascha", "buyins": 1, guest: false, "payout": 7.80, "result": 2.70}}
    ]
    """

    # --- CALL MODEL ---
    logger.info("Calling model %s with image (size=%d bytes)", MODEL_NAME, len(resized_bytes))
    start_model = time.time()
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=[prompt, image_part])
    except Exception as e:
        logger.exception("Model call failed: %s", e)
        raise
    model_duration = time.time() - start_model

    raw_text = response.text.strip()
    logger.info("Model response length=%d duration=%.3fs", len(raw_text), model_duration)

    try:
        data = extract_json(raw_text)
    except Exception as e:
        logger.exception("Failed to parse JSON from model output: %s", e)
        raise
    logger.info("Parsed JSON rows=%d", len(data) if isinstance(data, list) else 0)

    cleaned = []
    for row in data:
        logger.info("Raw row: %s", row)
        fixed = fix_name(row.get("name", ""), names)
        if fixed:
            cleaned_row = {
                "name": fixed,
                "guest": bool(row.get("guest", False)),
                "buyins": int(row.get("buyins", 0)),
                "payout": float(row.get("payout", 0)),
                "result": float(row.get("result", 0))
            }
            logger.info("Cleaned row: %s", cleaned_row)
            cleaned.append(cleaned_row)

    return {"results": cleaned}