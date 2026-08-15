import os
import re
import requests
import json
import logging
import time
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# =========================================================================
# PROMPTS
# =========================================================================

from prompts import Base_prompt, BASEREVIEWER_PROMPT

# =========================================================================
# CONFIG
# =========================================================================

# No environment key fallbacks – everything comes from the request.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "qwen/qwen3.6-27b"   # or "llama-3.3-70b-versatile"

REVIEWER_SYSTEM_PROMPT = (
    "You are a senior Roblox Luau code reviewer. Output ONLY raw "
    "[SCRIPT_START]...[SCRIPT_END] blocks per the format given in the "
    "user prompt, or 'No Changes Required. [Reviewed: ...]' if nothing "
    "needed fixing. No markdown, no extra prose."
)

logging.basicConfig(level=logging.INFO)

SCRIPT_BLOCK_RE = re.compile(r"\[SCRIPT_START\](.*?)\[SCRIPT_END\]", re.DOTALL)
SCRIPT_NAME_RE = re.compile(r"scriptName\s*=\s*(.+)")

# =========================================================================
# HELPERS
# =========================================================================

def truncate_text(text, max_chars=20000):
    """Truncate text to roughly stay within token limits (1 token ~ 4 chars)."""
    if not text:
        return text
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n... [TRUNCATED due to Groq API token limits]"
    return text

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

def call_gemini(model, prompt, api_key, max_retries=3, timeout=(10, 1200)):
    full_prompt = (
        "You are a code generator. You MUST output ONLY raw [SCRIPT_START]...[SCRIPT_END] blocks. "
        "NO explanations, NO markdown, NO reasoning, NO extra text. "
        "If no server script is needed, output exactly: 'No Server Script. [Script: <ScriptName>]'.\n\n"
        + prompt
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 16384,
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
            app.logger.warning(f"Timeout on attempt {attempt+1}/{max_retries}, extending timeout...")
            timeout = (timeout[0], timeout[1] + 120)
            time.sleep(2 ** attempt)
        except Exception as e:
            app.logger.error(f"Request exception: {e}")
            break
    raise Exception(f"All retries failed for model {model}")

def call_groq_review(prompt, api_key, model=GROQ_MODEL, max_retries=2, timeout=180.0):
    """Calls Groq's OpenAI-compatible API for a single review request."""
    client = OpenAI(
        api_key=api_key,
        base_url=GROQ_BASE_URL,
        timeout=timeout,
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                extra_body={
                    "reasoning_format": "hidden",   # hides reasoning output
                    "reasoning_effort": "none"      # disables thinking altogether
                }
            )
            return response.choices[0].message.content
        except Exception as e:
            app.logger.warning(f"Groq review attempt {attempt+1}/{max_retries} failed: {e}")
            last_error = e
            time.sleep(2 ** attempt)

    raise last_error

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
        gemini_api_key = data.get("api_key")          # Gemini key from request
        gemini_model = data.get("ai_model")           # Gemini model from request
        groq_api_key = data.get("review_key")         # Groq key from request (optional)

        # --- Validate required fields ---
        if not clientscript:
            return jsonify({"error": "Missing 'client_scripts' field"}), 400
        if not gemini_api_key:
            return jsonify({"error": "Missing 'api_key' for Gemini"}), 400
        if not gemini_model:
            return jsonify({"error": "Missing 'ai_model' field"}), 400

        FormattedPrompt = Base_prompt.format(
            first=clientscript,
            second=additional or "None provided.",
            third=serverside or "None provided.",
        )

        # --- Generation using Gemini ---
        app.logger.info(f"Sending generation request to Gemini with model: {gemini_model}")
        try:
            gen_resp = call_gemini(gemini_model, FormattedPrompt, gemini_api_key, max_retries=3, timeout=(10, 1200))
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

        # --- Review (Groq only) - SIMPLE TWO-HALF SPLIT ---
        final_output = output_text
        changed_scripts = []
        review_raw = None

        # If Groq key is provided, run the review; otherwise skip
        if groq_api_key:
            # Truncate client_scripts to avoid token limit (optional)
            truncated_clientscript = truncate_text(clientscript, max_chars=20000)

            # Split the server_output into two roughly equal halves
            total_len = len(output_text)
            half = total_len // 2
            # Try to split on newline to avoid cutting in the middle of a line
            split_pos = output_text.find('\n', half)
            if split_pos == -1:
                split_pos = half
            first_half = output_text[:split_pos]
            second_half = output_text[split_pos:]

            app.logger.info(f"Split output into two halves: {len(first_half)} and {len(second_half)} chars")

            all_reviewed_blocks = {}
            all_changed = []
            review_raw_parts = []

            # Review first half
            try:
                prompt1 = BASEREVIEWER_PROMPT.format(
                    client_scripts=truncated_clientscript,
                    server_output=first_half,
                    pre_existing_scripts=serverside or "None provided.",
                )
                review1 = call_groq_review(prompt1, groq_api_key)
                review1_blocks = parse_script_blocks(review1)
                all_reviewed_blocks.update(review1_blocks)
                all_changed.extend(review1_blocks.keys())
                review_raw_parts.append(f"--- FIRST HALF ---\n{review1}")
            except Exception as e:
                app.logger.error(f"First half review failed: {e}, keeping original")
                orig_blocks1 = parse_script_blocks(first_half)
                all_reviewed_blocks.update(orig_blocks1)

            # Small delay to avoid rate limiting
            time.sleep(1.5)

            # Review second half
            try:
                prompt2 = BASEREVIEWER_PROMPT.format(
                    client_scripts=truncated_clientscript,
                    server_output=second_half,
                    pre_existing_scripts=serverside or "None provided.",
                )
                review2 = call_groq_review(prompt2, groq_api_key)
                review2_blocks = parse_script_blocks(review2)
                all_reviewed_blocks.update(review2_blocks)
                all_changed.extend(review2_blocks.keys())
                review_raw_parts.append(f"--- SECOND HALF ---\n{review2}")
            except Exception as e:
                app.logger.error(f"Second half review failed: {e}, keeping original")
                orig_blocks2 = parse_script_blocks(second_half)
                all_reviewed_blocks.update(orig_blocks2)

            # Merge reviewed blocks with original blocks
            original_blocks = parse_script_blocks(output_text)
            merged_blocks = dict(original_blocks)
            merged_blocks.update(all_reviewed_blocks)  # reviewed overrides original
            final_output = "\n\n".join(merged_blocks.values())
            changed_scripts = list(all_changed)
            review_raw = "\n\n".join(review_raw_parts)

        return jsonify({
            "response": final_output,
            "generation_raw": output_text,
            "review_raw": review_raw,
            "review_model_used": GROQ_MODEL if groq_api_key else None,
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
