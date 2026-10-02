
import os
import json
import time

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app)

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


@app.route("/")
def home():
    return send_from_directory(".", "sonder.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}

    message = data.get("message", "")
    history = data.get("history", [])

    prompt = f"""
You are the cognitive analysis engine for a journaling application.

Analyze the user's message and return ONLY valid JSON.

Required fields:
- emotion: one main emotion
- confidence: number from 0 to 1
- intensity: Low, Medium, or High
- intent: short description of the user's intent
- tone: short description of tone
- priority: Emotion First, Action First, or Balanced
- response: a short, empathetic response to the user

Conversation history:
{json.dumps(history)}

User message:
{message}
"""

    models_to_try = [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest"
    ]

    for model_name in models_to_try:

        for attempt in range(2):

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.7
                    )
                )

                result = json.loads(response.text)

                if result and result.get("emotion"):
                    return jsonify(result)

            except Exception as e:
                print(
                    f"{model_name} attempt "
                    f"{attempt + 1} failed: {e}"
                )
                time.sleep(1.5)

    return jsonify({
        "emotion": "neutral",
        "confidence": 0.4,
        "intensity": "Low",
        "intent": "General",
        "tone": "Plain",
        "priority": "Balanced",
        "response": "The analysis service is temporarily unavailable. Please try again."
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
