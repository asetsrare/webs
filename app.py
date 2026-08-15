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
GROQ_MODEL = "qwen/qwen3.6-27b"   # or "llama-3.3-70b-versatile"
MAX_TOKENS_PER_REQUEST = 4000     # stay well below the 8000 TPM limit (leaves room for overhead)

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

def truncate_text(text, max_chars=10000):
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
                    "reasoning_format": "hidden",
                    "reasoning_effort": "none"
                }
            )
            return response.choices[0].message.content
        except Exception as e:
            app.logger.warning(f"Groq review attempt {attempt+1}/{max_retries} failed: {e}")
            last_error = e
            time.sleep(2 ** attempt)
    raise last_error

def split_into_chunks(server_output, client_scripts, pre_existing, max_tokens=MAX_TOKENS_PER_REQUEST):
    """
    Split server_output into chunks so that each request stays under max_tokens.
    Accounts for overhead (client_scripts, pre_existing, prompt template).
    Returns a list of chunks (strings).
    """
    # Estimate overhead characters from the prompt template and fixed parts
    # The template itself adds ~500 chars, plus the client_scripts and pre_existing.
    TEMPLATE_OVERHEAD = 500
    overhead_chars = len(client_scripts) + len(pre_existing) + TEMPLATE_OVERHEAD

    # Maximum characters allowed for server_output per chunk
    # Use 3.5 chars per token to be conservative (especially for code with symbols)
    max_chars_per_chunk = int((max_tokens * 3.5) - overhead_chars)

    if max_chars_per_chunk <= 0:
        # Even the overhead alone exceeds the limit – we need to truncate further.
        raise ValueError("Overhead too large for token limit. Please reduce client_scripts or pre_existing size.")

    # If the entire output fits, return it as one chunk
    if len(server_output) <= max_chars_per_chunk:
        return [server_output]

    chunks = []
    # Try to split on newline boundaries first
    lines = server_output.splitlines(keepends=True)
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) > max_chars_per_chunk and current_chunk:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += line
    if current_chunk:
        chunks.append(current_chunk)

    # If any chunk is still too large (e.g., a single massive line), force-split by character
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chars_per_chunk:
            # Force split
            for i in range(0, len(chunk), max_chars_per_chunk):
                final_chunks.append(chunk[i:i+max_chars_per_chunk])
        else:
            final_chunks.append(chunk)

    return final_chunks

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
        gemini_api_key = data.get("api_key")
        gemini_model = data.get("ai_model")
        groq_api_key = data.get("review_key")

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

        # --- Generation ---
        app.logger.info(f"Sending generation request to Gemini with model: {gemini_model}")
        try:
            gen_resp = call_gemini(gemini_model, FormattedPrompt, gemini_api_key, max_retries=3, timeout=(10, 1200))
        except Exception as e:
            app.logger.error(f"Generation failed: {e}")
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

        # --- Review (Groq only) - DYNAMIC CHUNKING ---
        final_output = output_text
        changed_scripts = []
        review_raw = None

        if groq_api_key:
            # Aggressively truncate client_scripts and serverside to reduce overhead
            truncated_clientscript = truncate_text(clientscript, max_chars=10000)
            truncated_serverside = truncate_text(serverside or "", max_chars=5000)

            try:
                chunks = split_into_chunks(
                    server_output=output_text,
                    client_scripts=truncated_clientscript,
                    pre_existing=truncated_serverside,
                    max_tokens=MAX_TOKENS_PER_REQUEST
                )
            except ValueError as e:
                app.logger.error(f"Chunking failed: {e}")
                return jsonify({
                    "response": output_text,
                    "generation_raw": output_text,
                    "review_raw": None,
                    "review_model_used": None,
                    "scripts_changed_by_review": [],
                    "warning": "Review skipped because input too large even after truncation."
                })

            app.logger.info(f"Split output into {len(chunks)} chunks.")

            all_reviewed_blocks = {}
            all_changed = []
            review_raw_parts = []

            for idx, chunk in enumerate(chunks):
                try:
                    prompt = BASEREVIEWER_PROMPT.format(
                        client_scripts=truncated_clientscript,
                        server_output=chunk,
                        pre_existing_scripts=truncated_serverside or "None provided.",
                    )
                    review = call_groq_review(prompt, groq_api_key)
                    review_blocks = parse_script_blocks(review)
                    if review_blocks:
                        all_reviewed_blocks.update(review_blocks)
                        all_changed.extend(review_blocks.keys())
                    review_raw_parts.append(f"--- CHUNK {idx+1}/{len(chunks)} ---\n{review}")
                except Exception as e:
                    app.logger.error(f"Review failed for chunk {idx+1}: {e}, keeping original")
                    # Fallback: keep original blocks from this chunk
                    orig_blocks = parse_script_blocks(chunk)
                    all_reviewed_blocks.update(orig_blocks)

                # Delay between chunks to avoid rate limits
                if idx < len(chunks) - 1:
                    time.sleep(1.5)

            # Merge reviewed blocks with original blocks
            original_blocks = parse_script_blocks(output_text)
            merged_blocks = dict(original_blocks)
            merged_blocks.update(all_reviewed_blocks)
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
        app.logger.error(f"KeyError: {e}")
        return jsonify({"error": "Unexpected response structure", "details": str(e)}), 500
    except Exception as e:
        app.logger.error(f"Unhandled exception: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
