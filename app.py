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

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "qwen/qwen3.6-27b"
MAX_TOKENS_PER_REQUEST = 2500
CHUNK_DELAY = 30

CLIENT_MAX_CHARS = 8000
PREEXISTING_MAX_CHARS = 3000

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

def truncate_text(text, max_chars):
    if not text:
        return text
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n... [TRUNCATED]"
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
    for step in result.get("steps", []):
        if step.get("type") == "model_output":
            content = step.get("content", [])
            texts = [item.get("text", "") for item in content if item.get("type") == "text"]
            if texts:
                return " ".join(texts)
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
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 16384, "topP": 1.0}
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
            app.logger.warning(f"Timeout attempt {attempt+1}, extending...")
            timeout = (timeout[0], timeout[1] + 120)
            time.sleep(2 ** attempt)
        except Exception as e:
            app.logger.error(f"Request exception: {e}")
            break
    raise Exception(f"All retries failed for model {model}")

def call_groq_review(prompt, api_key, model=GROQ_MODEL, max_retries=2, timeout=180.0):
    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, timeout=timeout)
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
                extra_body={"reasoning_format": "hidden", "reasoning_effort": "none"}
            )
            return response.choices[0].message.content
        except Exception as e:
            app.logger.warning(f"Groq review attempt {attempt+1}/{max_retries} failed: {e}")
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise last_error

def split_into_chunks(server_output, client_scripts, pre_existing, max_tokens=MAX_TOKENS_PER_REQUEST):
    TEMPLATE_OVERHEAD = 500
    overhead_chars = len(client_scripts) + len(pre_existing) + TEMPLATE_OVERHEAD
    max_chars_per_chunk = int((max_tokens * 3.5) - overhead_chars)

    if max_chars_per_chunk <= 0:
        raise ValueError(f"Overhead ({overhead_chars} chars) exceeds token limit; need to truncate further.")

    if len(server_output) <= max_chars_per_chunk:
        return [server_output]

    chunks = []
    lines = server_output.splitlines(keepends=True)
    current = ""
    for line in lines:
        if len(current) + len(line) > max_chars_per_chunk and current:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)

    final = []
    for ch in chunks:
        if len(ch) > max_chars_per_chunk:
            for i in range(0, len(ch), max_chars_per_chunk):
                final.append(ch[i:i+max_chars_per_chunk])
        else:
            final.append(ch)
    return final

# =========================================================================
# ROUTES
# =========================================================================

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/generate", methods=["POST"])
def generate():
    # Base response structure
    response_data = {
        "success": False,
        "response": None,
        "generation_raw": None,
        "review_raw": None,
        "review_model_used": None,
        "scripts_changed_by_review": [],
        "error": None,
        "warning": None
    }

    try:
        data = request.get_json()
        if not data:
            response_data["error"] = "Missing JSON body"
            return jsonify(response_data), 400  # Still 400 for malformed request

        clientscript = data.get("client_scripts")
        additional = data.get("additional_info")
        serverside = data.get("server_scripts")
        gemini_api_key = data.get("api_key")
        gemini_model = data.get("ai_model")
        groq_api_key = data.get("review_key")

        # Validate required fields
        if not clientscript:
            response_data["error"] = "Missing 'client_scripts'"
            return jsonify(response_data), 400
        if not gemini_api_key:
            response_data["error"] = "Missing 'api_key' for Gemini"
            return jsonify(response_data), 400
        if not gemini_model:
            response_data["error"] = "Missing 'ai_model'"
            return jsonify(response_data), 400

        FormattedPrompt = Base_prompt.format(
            first=clientscript,
            second=additional or "None provided.",
            third=serverside or "None provided.",
        )

        # --- Generation ---
        app.logger.info(f"Generating with Gemini: {gemini_model}")
        try:
            gen_resp = call_gemini(gemini_model, FormattedPrompt, gemini_api_key)
        except Exception as e:
            response_data["error"] = f"Generation failed: {str(e)}"
            # Still return 200 so client doesn't see a 500, but success is false
            return jsonify(response_data), 200

        if gen_resp.status_code != 200:
            try:
                error_json = gen_resp.json()
                error_msg = error_json.get("error", {}).get("message", "Unknown")
            except:
                error_msg = gen_resp.text[:200]
            response_data["error"] = f"Gemini API error: {error_msg}"
            return jsonify(response_data), 200

        result = gen_resp.json()
        output_text = clean_model_output(extract_gemini_output(result))

        if not output_text or not output_text.strip():
            response_data["error"] = "Empty output from generation"
            return jsonify(response_data), 200

        # Generation succeeded
        response_data["success"] = True
        response_data["generation_raw"] = output_text
        response_data["response"] = output_text  # start with generation as final

        # --- Review (Groq) with chunking and rate-limit delay ---
        if groq_api_key:
            truncated_clientscript = truncate_text(clientscript, max_chars=CLIENT_MAX_CHARS)
            truncated_serverside = truncate_text(serverside or "", max_chars=PREEXISTING_MAX_CHARS)

            try:
                chunks = split_into_chunks(
                    server_output=output_text,
                    client_scripts=truncated_clientscript,
                    pre_existing=truncated_serverside,
                    max_tokens=MAX_TOKENS_PER_REQUEST
                )
            except ValueError as e:
                response_data["warning"] = f"Review skipped: {str(e)}"
                # Keep response as generation only
                return jsonify(response_data), 200

            app.logger.info(f"Reviewing {len(chunks)} chunks (delay {CHUNK_DELAY}s between)")

            all_reviewed_blocks = {}
            all_changed = []
            review_parts = []

            for idx, chunk in enumerate(chunks):
                try:
                    prompt = BASEREVIEWER_PROMPT.format(
                        client_scripts=truncated_clientscript,
                        server_output=chunk,
                        pre_existing_scripts=truncated_serverside or "None provided.",
                    )
                    review = call_groq_review(prompt, groq_api_key)
                    blocks = parse_script_blocks(review)
                    if blocks:
                        all_reviewed_blocks.update(blocks)
                        all_changed.extend(blocks.keys())
                    review_parts.append(f"--- CHUNK {idx+1}/{len(chunks)} ---\n{review}")
                except Exception as e:
                    app.logger.error(f"Chunk {idx+1} failed: {e}, keeping original")
                    orig_blocks = parse_script_blocks(chunk)
                    all_reviewed_blocks.update(orig_blocks)

                if idx < len(chunks) - 1:
                    app.logger.info(f"Waiting {CHUNK_DELAY}s before next chunk...")
                    time.sleep(CHUNK_DELAY)

            # Merge reviewed blocks with original
            original_blocks = parse_script_blocks(output_text)
            merged = dict(original_blocks)
            merged.update(all_reviewed_blocks)
            final_output = "\n\n".join(merged.values())
            response_data["response"] = final_output
            response_data["review_raw"] = "\n\n".join(review_parts)
            response_data["review_model_used"] = GROQ_MODEL
            response_data["scripts_changed_by_review"] = all_changed

        # If no review, we already have response set to generation
        return jsonify(response_data), 200

    except Exception as e:
        app.logger.error(f"Unhandled exception: {e}", exc_info=True)
        response_data["error"] = f"Internal server error: {str(e)}"
        return jsonify(response_data), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
