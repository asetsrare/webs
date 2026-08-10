import os
import re
import requests
import json
import logging
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

# =========================================================================
# PROMPTS
# =========================================================================

from prompts import Base_prompt, BASEREVIEWER_PROMPT

# =========================================================================
# CONFIG
# =========================================================================

SERVER_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
REVIEW_MODEL = "gemini-2.5-flash"   # or "gemini-3.5-flash" when available

logging.basicConfig(level=logging.INFO)

SCRIPT_BLOCK_RE = re.compile(r"\[SCRIPT_START\](.*?)\[SCRIPT_END\]", re.DOTALL)
SCRIPT_NAME_RE = re.compile(r"scriptName\s*=\s*(.+)")

# =========================================================================
# HELPERS
# =========================================================================

def clean_model_output(text):
    if not text:
        return text
    text = text.strip()
    text = re.sub(r"^```(?:lua)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = re.sub(r"^`\s*", "", text)
    text = re.sub(r"`\s*$", "", text)

    start_idx = text.find("[SCRIPT_START]")
    no_changes_idx = text.find("No Changes Required")
    no_server_idx = text.find("No Server Script")

    candidates = [i for i in (start_idx, no_changes_idx, no_server_idx) if i > 0]
    if candidates:
        text = text[min(candidates):]
    return text.strip()

def parse_script_blocks(text):
    blocks = {}
    if not text:
        return blocks
    for match in SCRIPT_BLOCK_RE.finditer(text):
        body = match.group(1)
        name_match = SCRIPT_NAME_RE.search(body)
        if not name_match:
            continue
        name = name_match.group(1).strip()
        blocks[name] = "[SCRIPT_START]" + body + "[SCRIPT_END]"
    return blocks

def merge_reviewed_output(base_text, review_text):
    review_text = clean_model_output(review_text)

    if not review_text or review_text.startswith("No Changes Required"):
        return base_text, []

    base_blocks = parse_script_blocks(base_text)
    review_blocks = parse_script_blocks(review_text)

    if not review_blocks:
        app.logger.warning("Review output not in [SCRIPT_START] format; keeping base output only")
        return base_text, []

    if not base_blocks:
        app.logger.warning("Base output not in [SCRIPT_START] format; skipping merge, returning both raw")
        return base_text + "\n\n" + review_text, list(review_blocks.keys())

    merged = dict(base_blocks)
    changed_names = list(review_blocks.keys())
    merged.update(review_blocks)
    return "\n\n".join(merged.values()), changed_names

def extract_gemini_output(result):
    # interactions style
    for step in result.get("steps", []):
        if step.get("type") == "model_output":
            content = step.get("content", [])
            texts = [item.get("text", "") for item in content if item.get("type") == "text"]
            if texts:
                return " ".join(texts)

    # generateContent style
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        pass

    return str(result)

def call_gemini(model, prompt, api_key, max_retries=2, timeout=(10, 600)):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 4096,
            "topP": 1.0,
        }
    }
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code >= 400:
                app.logger.error(f"Gemini API error (status {resp.status_code}): {resp.text[:200]}")
                return resp
        except requests.Timeout:
            app.logger.warning(f"Request timed out on attempt {attempt+1}/{max_retries}, retrying...")
            time.sleep(2 ** attempt)
        except Exception as e:
            app.logger.error(f"Request exception: {e}")
            break
    raise Exception(f"All retries failed for model {model}")

# =========================================================================
# ROUTES
# =========================================================================

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        clientscript = data.get("client_scripts")
        additional = data.get("additional_info")
        serverside = data.get("server_scripts")

        if not clientscript:
            return jsonify({"error": "Missing 'client_scripts' field"}), 400

        FormattedPrompt = Base_prompt.format(
            first=clientscript,
            second=additional or "None provided.",
            third=serverside or "None provided.",
        )

        client_api_key = data.get("api_key")
        client_ai_model = data.get("ai_model")

        api_key_to_use = client_api_key or SERVER_API_KEY
        if not api_key_to_use:
            return jsonify({"error": "No API key provided. Please enter your API key in the plugin."}), 400
        if not client_ai_model:
            return jsonify({"error": "Missing 'ai_model' field"}), 400

        # --- Generation ---
        app.logger.info(f"Sending generation request to Gemini with model: {client_ai_model}")
        try:
            gen_resp = call_gemini(client_ai_model, FormattedPrompt, api_key_to_use, max_retries=2, timeout=(10, 300))
        except Exception as e:
            app.logger.error(f"Generation failed after retries: {e}")
            return jsonify({"error": "Generation failed", "details": str(e)}), 500

        if gen_resp.status_code != 200:
            try:
                error_json = gen_resp.json()
                error_msg = error_json.get("error", {}).get("message", "Unknown error")
                error_details = json.dumps(error_json)
            except Exception:
                error_msg = gen_resp.text[:200]
                error_details = gen_resp.text
            app.logger.error(f"Gemini API error: {error_msg}")
            return jsonify({"error": "Gemini API error", "details": error_msg, "raw": error_details}), 500

        result = gen_resp.json()
        output_text = clean_model_output(extract_gemini_output(result))

        if output_text is None or not output_text.strip():
            return jsonify({"error": "Empty output from generation model"}), 500

        # --- Review ---
        review_result = None
        final_output = output_text
        changed_scripts = []

        try:
            FormattedReviewPrompt = BASEREVIEWER_PROMPT.format(
                client_scripts=clientscript,
                server_output=output_text,
                pre_existing_scripts=serverside or "None provided.",
            )

            app.logger.info("Sending review request to Gemini...")
            review_resp = call_gemini(REVIEW_MODEL, FormattedReviewPrompt, api_key_to_use, max_retries=3, timeout=(10, 600))

            if review_resp.status_code == 200:
                review_json = review_resp.json()
                review_message = extract_gemini_output(review_json)
                review_result = clean_model_output(review_message)
                final_output, changed_scripts = merge_reviewed_output(output_text, review_result)
            else:
                app.logger.error(f"Review API call failed with status {review_resp.status_code}: {review_resp.text[:200]}")
                final_output = output_text

        except Exception as e:
            app.logger.error(f"Review stage failed, returning unreviewed output: {e}")
            final_output = output_text

        return jsonify({
            "response": final_output,
            "generation_raw": output_text,
            "review_raw": review_result,
            "review_reasoning": None,
            "scripts_changed_by_review": changed_scripts,
        })

    except requests.exceptions.RequestException as e:
        app.logger.error(f"RequestException: {e}")
        return jsonify({"error": "Network/Request error", "details": str(e)}), 500
    except KeyError as e:
        app.logger.error(f"KeyError: {e} - response structure unexpected")
        return jsonify({"error": "Unexpected response structure", "details": str(e)}), 500
    except Exception as e:
        app.logger.error(f"Unhandled exception: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
