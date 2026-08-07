import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)
API_KEY = os.environ.get("GEMINI_API_KEY")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt'"}), 400

    payload = {"contents": [{"parts": [{"text": data["prompt"]}]}]}
    response = requests.post(URL, json=payload)
    
    if response.status_code != 200:
        return jsonify({"error": "Gemini API error", "details": response.text}), response.status_code

    result = response.json()
    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return jsonify({"response": text})
    except (KeyError, IndexError):
        return jsonify({"error": "Unexpected API response", "raw": result}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
