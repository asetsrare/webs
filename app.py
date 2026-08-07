import os
from flask import Flask, request, jsonify
from google import genai

app = Flask(__name__)

# Client reads API key from GOOGLE_API_KEY env var
client = genai.Client()

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
            environment="remote",
        )
        return jsonify({
            "interaction_id": interaction.id,
            "environment_id": interaction.environment_id,
            "response": interaction.output_text
        })
    except Exception as e:
        app.logger.error(f"GenAI error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
