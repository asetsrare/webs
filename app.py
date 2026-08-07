import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("No API key set. Set GOOGLE_API_KEY or GEMINI_API_KEY.")

# The managed agent name from your curl example
AGENT = "antigravity-preview-05-2026"
INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt'"}), 400

    # Build the payload for a new interaction (no previous_interaction_id)
    payload = {
        "agent": AGENT,
        "input": data["prompt"],
        "environment": "remote",
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY,
    }

    try:
        resp = requests.post(INTERACTIONS_URL, json=payload, headers=headers)
        resp.raise_for_status()
        result = resp.json()
        # The response structure: we expect "output_text" or similar
        output = result.get("output_text") or result.get("output") or result
        return jsonify({"response": output})
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Interactions API error: {e}")
        # Return the error details from Google if possible
        error_detail = e.response.text if e.response else str(e)
        return jsonify({"error": error_detail}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
