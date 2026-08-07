import os
import json
import google.generativeai as genai
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configure Gemini with your API key
# On Render, set the environment variable GEMINI_API_KEY
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=API_KEY)

# Choose a model – "gemini-2.0-flash" is fast and good
MODEL_NAME = "gemini-3.5-flash"  # or "gemini-1.5-pro" etc.

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Gemini API service is running"})

@app.route("/generate", methods=["POST"])
def generate():
    """
    Expects JSON: { "prompt": "your prompt here" }
    Returns: { "response": "gemini output" }
    """
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' field"}), 400

    user_prompt = data["prompt"]

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(user_prompt)
        # Extract the text from the response
        if response.text:
            return jsonify({"response": response.text})
        else:
            # Sometimes the response is blocked or empty
            return jsonify({"error": "Empty response from Gemini", "details": response.prompt_feedback}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local development only – Render uses Gunicorn
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
