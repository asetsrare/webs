import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Server's own key (optional fallback)
SERVER_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

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

    # Get the API key from the client (plugin)
    client_api_key = data.get("api_key")
    
    # Use client's key if provided, otherwise fallback to server's key
    api_key_to_use = client_api_key or SERVER_API_KEY
    
    if not api_key_to_use:
        return jsonify({"error": "No API key provided. Please enter your API key in the plugin."}), 400

    payload = {
        "agent": AGENT,
        "input": data["prompt"],
        "environment": "remote",
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key_to_use,
    }

    try:
        resp = requests.post(INTERACTIONS_URL, json=payload, headers=headers)
        resp.raise_for_status()
        result = resp.json()

        # Extract the final output
        output_text = None
        for step in result.get("steps", []):
            if step.get("type") == "model_output":
                content = step.get("content", [])
                texts = [item.get("text", "") for item in content if item.get("type") == "text"]
                output_text = " ".join(texts)
                break
        
        if output_text is None:
            output_text = result.get("output_text") or result.get("output") or str(result)

        return jsonify({"response": output_text})
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Interactions API error: {e}")
        error_detail = e.response.text if e.response else str(e)
        return jsonify({"error": error_detail}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
