import os
from flask import Flask, request, jsonify
from google import genai

app = Flask(__name__)
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

AGENT = "antigravity-preview-05-2026"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt'"}), 400

    try:
        interaction = client.interactions.create(
            agent=AGENT,
            input=data["prompt"],
            environment="remote",
        )
        return jsonify({"response": interaction.output_text})
    except Exception as e:
        app.logger.error(f"GenAI error: {e}")
        return jsonify({"error": str(e)}), 500
