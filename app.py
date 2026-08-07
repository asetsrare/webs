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

        # Extract the final output from the steps
        output_text = None
        for step in result.get("steps", []):
            if step.get("type") == "model_output":
                content = step.get("content", [])
                # Concatenate all text pieces
                texts = [item.get("text", "") for item in content if item.get("type") == "text"]
                output_text = " ".join(texts)
                break
        
        # Fallback if no model_output step found
        if output_text is None:
            output_text = result.get("output_text") or result.get("output") or str(result)

        return jsonify({"response": output_text})
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Interactions API error: {e}")
        error_detail = e.response.text if e.response else str(e)
        return jsonify({"error": error_detail}), 500
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
