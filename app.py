import os
import requests
import json
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)

# Server's own key (optional fallback)
SERVER_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

# Set up logging (Render will capture this)
logging.basicConfig(level=logging.INFO)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/generate", methods=["POST"])
def generate():
    try:
        # 1. Parse request
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        if "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' field"}), 400

        # 2. Get API key and agent name
        client_api_key = data.get("api_key")
        client_ai_model = data.get("ai_model")  

        # Use client's key if provided, otherwise fallback to server's key
        api_key_to_use = client_api_key or SERVER_API_KEY
        if not api_key_to_use:
            return jsonify({"error": "No API key provided. Please enter your API key in the plugin."}), 400

        # 3. Build payload for Google
        payload = {
            "model": client_ai_model,
            "input": data["prompt"],
            "environment": "remote",
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key_to_use,
        }

        # 4. Send request to Google with timeout
        app.logger.info(f"Sending request to Google with agent: {client_ai_model}")
        resp = requests.post(INTERACTIONS_URL, json=payload, headers=headers, timeout=100)
        app.logger.info(f"Google responded with status: {resp.status_code}")

        # 5. Check for HTTP errors from Google
        if resp.status_code != 200:
            # Try to parse error JSON
            try:
                error_json = resp.json()
                error_msg = error_json.get("error", {}).get("message", "Unknown error from Google")
                error_details = json.dumps(error_json)
            except:
                error_msg = resp.text[:200]  # Truncate
                error_details = resp.text
            app.logger.error(f"Google API error: {error_msg}")
            return jsonify({"error": "Google API error", "details": error_msg, "raw": error_details}), 500

        # 6. Parse successful response
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
        app.logger.error(f"RequestException: {e}")
        return jsonify({"error": "Network/Request error", "details": str(e)}), 500
    except KeyError as e:
        app.logger.error(f"KeyError: {e} - response structure unexpected")
        return jsonify({"error": "Unexpected response structure from Google", "details": str(e)}), 500
    except Exception as e:
        app.logger.error(f"Unhandled exception: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
