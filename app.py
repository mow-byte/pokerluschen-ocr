from google import genai
from google.genai.types import Part
from fastapi import FastAPI, UploadFile, File, Query
import json
from typing import List
from difflib import get_close_matches
import os

app = FastAPI()


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
async def extract(file: UploadFile = File(...), names: List[str] = Query(...)):
    image_bytes = await file.read()

    image_part = Part.from_bytes(data=image_bytes, mime_type=file.content_type)

    # --- PROMPT ---
    prompt = f"""
    You are extracting structured data from a handwritten scorecard.

    Players must ONLY be from this list:
    {names}

    Rules:
    - Do NOT invent names
    - If unclear, map to closest valid name
    - Instead of the name "Hannes", its nickname "Mow" is also valid. Report "Hannes" in this case
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
    print("Calling ai..")
    response = client.models.generate_content(model=MODEL_NAME, contents=[prompt, image_part])
    raw_text = response.text.strip()

    data = extract_json(raw_text)
    print("Result:")
    print(data)

    cleaned = []
    for row in data:
        fixed = fix_name(row.get("name", ""), names)
        if fixed:
            cleaned.append({
                "name": fixed,
                "guest": bool(row.get("guest", False)),
                "buyins": int(row.get("buyins", 0)),
                "payout": float(row.get("payout", 0)),
                "result": float(row.get("result", 0))
            })

    return {"results": cleaned}