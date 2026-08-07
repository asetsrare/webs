import os
from flask import Flask, request, jsonify
from google import genai

app = Flask(__name__)

# The client will look for the API key in the environment variable GOOGLE_API_KEY
# or uses Application Default Credentials if running on GCP.
# On Render, set the env var GOOGLE_API_KEY to your Gemini API key.
client = genai.Client()

# The agent name you want to use
AGENT = "antigravity-preview-05-2026"

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt'"}), 400

    try:
        interaction = client.interactions.create(
            agent=AGENT,
            input=data["prompt"],
            environment="remote",   # or "local" if you want to run it locally
        )
        # The output is available in interaction.output_text
        return jsonify({"response": interaction.output_text})
    except Exception as e:
        # Log the error (you can also print to console)
        app.logger.error(f"GenAI error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
