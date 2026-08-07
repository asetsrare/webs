import os
import json
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
    # Try to parse JSON with silent=True (won't raise exception)
    data = request.get_json(silent=True)
    
    # If that failed, try manually parsing from raw data
    if data is None:
        try:
            raw_data = request.data.decode('utf-8')
            data = json.loads(raw_data)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            return jsonify({
                "error": "Invalid JSON payload",
                "details": str(e),
                "received": request.data.decode('utf-8', errors='ignore')[:200]
            }), 400
    
    # Check if we have the 'prompt' field
    if not data or "prompt" not in data:
        return jsonify({
            "error": "Missing 'prompt' field",
            "received_keys": list(data.keys()) if data else []
        }), 400
    
    # Call Gemini API
    try:
        payload = {"contents": [{"parts": [{"text": data["prompt"]}]}]}
        response = requests.post(URL, json=payload)
        
        if response.status_code != 200:
            return jsonify({
                "error": "Gemini API error",
                "status": response.status_code,
                "details": response.text
            }), response.status_code
        
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return jsonify({"response": text})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
