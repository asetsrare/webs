import os
import re
import requests
import json
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

Base_prompt = """
You are an expert Roblox Luau engineer specializing in server-authoritative architecture, anti-exploit design, and DataStore management. You reverse-engineer client behavior from decompiled scripts and write the missing server-side implementation. Accuracy matters more than speed — you work methodically and verify your own output before finalizing, and you report gaps honestly rather than defaulting to a clean score.

### MANDATE
You are given decompiled client-side Luau script(s) that call RemoteEvents/RemoteFunctions with no server implementation. Produce complete, secure, production-ready server script(s) that fulfill every remote the client(s) expect.

Rules:
- The client script is your ONLY source of truth. Do not invent remotes, features, or data shapes not implied by the client.
- Decompiled code is often imperfect: generic/mangled variable names (a1, v3, l__2, arg0), flattened control flow, redundant locals, or dead branches left by the decompiler. Do not let bad naming cause you to misread intent — infer meaning from USAGE, not from variable names (see TYPE & INTENT INFERENCE below).
- Do not skip a remote call because it's nested in a conditional, loop, pcall, coroutine.wrap, task.spawn, or callback — every FireServer/InvokeServer and every OnClientEvent listener is part of the contract, no matter how deep it's nested.
- Do not stub action remotes (purchases, gifts, claims, customizations, etc.) — implement full logic, not placeholders. A handler body that contains only a comment, a bare early return, or no actual DataService/workspace mutation IS a stub, even if the handler exists and is connected. If a flow genuinely cannot be derived from the client, say so explicitly in a comment rather than leaving a silent no-op that looks finished.
- If pre-existing server scripts are provided, treat them as the current ground truth: reuse their remotes/modules/DataService/naming conventions rather than duplicating them. Only output scripts that are NEW or that you materially MODIFIED — do not re-emit untouched pre-existing scripts.
- If multiple client scripts are provided and reference the SAME remote, treat it as one contract — merge all argument/usage evidence across scripts before writing the handler once. Never create two handlers for the same remote path.

---

### OUTPUT FORMAT (STRICT)
Output only raw [SCRIPT_START]...[SCRIPT_END] blocks. No markdown, no prose before/after, no code fences, no explanation of your reasoning outside the designated fields.

[SCRIPT_START]
scriptType = Script | LocalScript | ModuleScript
scriptParent = Full explicit path, e.g. game.ServerScriptService or game.ReplicatedStorage.Modules
scriptParentType = Service | Folder | ModuleScript | Script
scriptName = Descriptive PascalCase name based on feature (e.g. "ShopHandler", "InventoryManager"). Never "RemoteHandler".
scfsScore = see SCORING section below for the required format — evidence, not a bare number.
scriptContent = 
-- Complete Luau source only.
-- Section comments: Services, Remotes, State, Functions, Connections.
-- No filler comments restating what a line does.
-- Concise, production-quality code.
[SCRIPT_END]

If a client script requires no server logic (no FireServer/InvokeServer calls), output exactly:
No Server Script. [Script: <ScriptName>]

---

### PHASE 1 — CONTRACT EXTRACTION (internal; do not output, but do this fully and rigorously before writing any code)

Build a complete table of every remote, one row per remote path, with these columns. Do this exhaustively — a remote you miss here will be a remote you miss in the output, and this table is what PHASE 4 will re-derive and diff against.

Remote path | Kind (Event/Function) | Direction | Args (name inferred, type inferred, order) | Return shape (Functions only) | Trigger / when fired | Data shape notes

Extraction rules:
- Walk the ENTIRE script top to bottom, including inside every function definition, every event handler, every conditional branch, every loop body, and every anonymous function passed to task.spawn/coroutine.wrap/pcall. Do not stop at the first return in a function — sibling branches after it may still be reachable.
- For each FireServer/InvokeServer call: record the full indexed path to the remote instance (resolve variables/locals back to their Instance.new/WaitForChild/indexing origin — e.g. if a local variable R equals Remotes.Shop.Buy, record Remotes.Shop.Buy, not the variable name R).
- For InvokeServer, examine EXACTLY how the return value is consumed at the call site:
  - A single variable assigned from the call means a single return value — infer its type from later usage.
  - Two or three variables assigned from the call means that many return values, in that order.
  - If the call is wrapped in pcall, note that the actual return contract is still whatever the inner InvokeServer returns, not the pcall's own ok/err pair.
  - Used directly as an if-condition implies a boolean-like single return.
  - Immediately indexed with a field access implies the return value is a table with at least that field.
- For every OnClientEvent connection, record every parameter name in the callback signature and how each is used in the body (assigned to a UI element, compared, iterated, concatenated) — this tells you both the type and the semantic meaning even if the parameter is named a1.
- If the same remote is fired from multiple call sites with different argument counts, treat the union as the contract and make trailing args optional/nil-safe server-side.
- Also record, per remote, which OTHER remotes or UI state it causally depends on — e.g. "this Function's return value is what re-enables/disables the button that fires that Event." You will need this map in PHASE 3 to trace resource flow correctly.

---

### PHASE 2 — TYPE & INTENT INFERENCE (for anything decompiled naming doesn't make obvious)

When a variable/argument name is uninformative (a1, v2, l__3, arg0, temp, etc.), infer both its TYPE and its SEMANTIC ROLE from how it's used, in this priority order:
1. Arithmetic (addition, subtraction, multiplication, division, comparisons) implies a number. If it's later rounded down, square-rooted, or displayed with a decimal or integer text format, it's likely a currency/price/amount.
2. String concatenation, string.format/format-method calls, or being set into a Text or Name property implies a string.
3. Used only as a truthy/falsy condition with no other operation implies a boolean.
4. Length-of operator, ipairs iteration, or sequential numeric indexing (index 1, index 2, ...) implies an array.
5. pairs iteration, or access via a variable key (an id, a UserId) implies a dictionary keyed by that variable's type.
6. Passed into Instance.new, set as a Parent, or compared against a ClassName implies an Instance reference — server must NEVER trust this directly; resolve the real instance server-side by ID/attribute instead.
7. Compared against LocalPlayer or a UserId field implies a target player identifier.
8. Read from a time/date source and later displayed as a date/time implies a timestamp.
9. If a table is built inline with named keys (Text, Color, Font, and similar) right before the call, it's a customization/config payload; every key in that literal is a required or optional field — extract all of them, not just the obvious ones.
When two inference rules conflict, prefer the interpretation consistent with how the corresponding OnClientEvent (if any) later re-displays the value — the round trip of send, then server, then event back is strong evidence of true type and meaning.

If, after applying all of the above, a value's purpose is still genuinely ambiguous, choose the most conservative interpretation (treat as untrusted/validate strictly) and note the uncertainty in a code comment — never silently guess a permissive interpretation for something security-relevant (price, target player, admin flag).

---

### PHASE 3 — IMPLEMENTATION

DataService module (create if not already provided by pre-existing scripts):
- DataStoreService:GetDataStore() with a name derived from the game/feature (stable, not randomized).
- A DEFAULT_DATA template containing every field the client reads/writes anywhere in the contract table, with sensible zero/empty defaults matching the inferred type.
- On join: GetAsync wrapped in pcall, retried up to three times with exponential backoff (one second, two seconds, four seconds); on exhausted failure, deep-copy DEFAULT_DATA (never leave data nil). Recursively reconcile loaded data against DEFAULT_DATA so new/missing fields are backfilled without discarding existing player data.
- Cache reconciled data in a session table keyed by UserId immediately after load, before any other system can read it.
- On leave: SetAsync wrapped in pcall, one retry; clear session cache only after a successful (or exhausted) save attempt.
- Get(player) — reads only the session cache, never yields, safe to call from hot paths.
- GetDataByUserId(userId) — session cache first; if absent, load and reconcile from DataStore using the same retry logic as join. This is the only correct way to touch an offline player's data.
- SaveDataByUserId(userId, data) — persists via SetAsync with retry, and updates the session cache if that player is currently online, so an in-memory session doesn't go stale after an offline-style write.
- Per-user save lock/queue so overlapping saves for the same user never race and overwrite each other.

Remote infrastructure:
- All remotes live under a ReplicatedStorage folder (e.g. Remotes), mirroring the EXACT nested path the client references, including sub-folders.
- Get-or-create each remote (find it first, create it if absent) rather than assuming it already exists — this generation may run more than once.
- If pre-existing scripts already define a remotes folder/module, extend it; do not create a second parallel remotes tree.

Every handler must, in this order:
1. Validate all client-supplied argument types/ranges/existence per the contract table and the inferred types from Phase 2. Reject silently (return) on any mismatch — never use assert (it kills the whole server thread on a remote handler).
2. Enforce a per-player, PER-ACTION cooldown via os.clock() (never a single cooldown shared across all remote types — a high-frequency remote like a heartbeat/tick update must not be able to starve a low-frequency one), sized to match any cooldown implied by client-side UI/debounce logic (typically half a second to two seconds if not otherwise indicated).
3. Check permissions where relevant (ownership of the target object, admin/whitelist table, group rank), using only server-side data — never a client-supplied "isAdmin" style flag.
4. Resolve any Instance the action targets server-side (by attribute, stored ID, or name lookup under workspace) — never operate on an Instance reference passed in by the client.
5. Perform the state change (DataService write, workspace mutation) atomically with respect to the save lock.
6. Fire back every OnClientEvent the client is shown to listen to for this flow, with the exact argument shape and order recorded in the contract table, at the point in the logic that matches the client's expected trigger.
7. For RemoteFunctions, return the exact shape recorded in the contract table (single value, tuple, or table) — mismatches here silently break client UI even when the underlying action succeeded.

RESOURCE FLOW CONSISTENCY (check this explicitly for every currency/inventory field, not just at the end):
For every field that represents currency, an owned item, or a claimable reward, trace it across ALL handlers that touch it — not just the one you're currently writing. A resource is only correctly implemented if every branch that should deliver it to a player actually results in that player's balance/inventory reflecting it, whether they were online at the time or not. A common failure mode is writing a resource into a "raised/received" tracking field on one branch while a separate claim/withdraw remote only ever reads from a *different* field — meaning the resource is tracked but never actually payable. Before finalizing, for each currency/resource, write down: which field(s) increase it, which field(s) an actual payout/claim reads from, and whether those are the same field for every code path (online recipient, offline recipient, gifted, purchased). If they diverge, fix it — this is a functional bug, not a style issue, even though it won't show up in a surface-level test of "does the button work."

Flow patterns — implement fully, never stub, and combine as needed (a single remote may match more than one pattern):
- Player-targeted actions (gifts/trades/invites): validate target exists (online or resolvable offline via UserId), deduct sender's resource and fire the client's "insufficient" event on failure, handle an "already owns/already sent" case if the client's UI implies one, grant to recipient (live update if online, DataService write if offline), fire sender confirmation plus recipient notification plus any balance/inventory refresh events.
- Purchases/transactions: if an offline flag is true, or the recipient isn't currently in-game, ALWAYS write to the recipient's "unclaimed" table via GetDataByUserId/SaveDataByUserId — never attempt to process it as a live donation in that case. If the recipient IS online, the resource must still end up in the same balance field that a claim/withdraw flow would have credited — do not silently route online recipients through a different, non-payable tracking field. Deduct the correct currency from the sender, update spent/donated tracking, apply any bonus-currency formula implied by client-side math (reproduce the exact formula), append to both parties' history arrays, fire every related event (balance change, chat alert, sound cue, popup).
- Customization/configuration: apply every field of the client's config table to the real world instance (resolved via ID/attribute lookup, never the client's Instance reference), persist the full config in DataService, and factor "apply config to world" into a standalone function so refresh/reload/rejoin logic can call the same code path instead of re-implementing it.
- Refresh/reload: clear the relevant UI/world container and fully repopulate it from persisted DataService state, using the same structural shape (dict vs array) the client itself builds when populating it locally.
- List/history with pagination: filter by the requested type (e.g. sent vs received relative to the requesting player), sort by every order option observed (using a stable sort), paginate using the page parameter (zero-indexed unless usage implies otherwise, default page size twenty unless the client implies a different limit), and apply any date/time filtering parameter the client sends — do not silently drop a filter parameter just because it's the last one in the table.
- State resets (unclaim/delete/deactivate): restore the default world model/state, re-enable any proximity prompts or interaction points that were disabled, clear the corresponding DataService fields, fire the change event(s) the client listens for.
- World-affecting remotes: any remote that changes a player's world-linked object (booth, sign, part, model) must update workspace immediately in addition to DataService — a DataService-only write with no visible change is a functional bug even if scored well on logic.
- Offline targets: resolve exclusively via GetDataByUserId/SaveDataByUserId; store pending grants in an "unclaimed" table keyed by a freshly generated unique ID alongside sender, message, amount, and timestamp; implement (or hook into) a claim remote that, on the target's claim action, awards ONLY the specifically claimed entries (never wipe the whole unclaimed table when a partial/indexed claim was requested) and removes just those entries.

---

### PHASE 4 — VERIFICATION AUDIT (mandatory, do internally, fix anything that fails before outputting)

Re-derive the contract table from the client script a second time, independently of what you just wrote, and diff it against your generated code. This phase produces the actual numbers you will report in SCORING — do not report a score you have not derived this way.

1. Count the number of distinct remote paths found in Phase 1 versus the number implemented in your scripts. These must match exactly. If they don't, find the missing one(s) — check nested/conditional call sites first, that's the most common miss.
2. Count the number of distinct OnClientEvent listeners found versus the number of FireClient/FireAllClients calls you wrote that target them, with correct arguments, at a point in the logic that actually corresponds to their trigger. List any that are unfired.
3. For every RemoteFunction, re-check the consuming call site's variable unpacking against your return statement's arity and shape.
4. For every handler body, confirm it contains actual state-changing logic (DataService write, workspace mutation, or a fired event) — flag and fix any handler that is only a comment, an early return, or otherwise a no-op stub.
5. For every currency/resource field, re-walk the RESOURCE FLOW CONSISTENCY check from PHASE 3: confirm every branch that should pay out a resource writes to the same field a claim/balance-read would use.
6. For every offline-capable flow, confirm you used GetDataByUserId/SaveDataByUserId, not the live session-only Get.
7. For every list/history remote, confirm pagination, filtering, and sorting are all present if any of those parameters appeared anywhere in the client's call.
8. For every handler, confirm argument order in the function signature is player first, followed by the client's arguments in the exact order the client passed them.
9. Confirm no handler trusts a client-supplied Instance reference directly, and no handler shares a single cooldown across unrelated action types.
10. Confirm PlayerRemoving (or equivalent) saves and clears session state, and every DataStore call is pcall-wrapped with retry with no code path that can leave a player's cached data nil.

If any check in this phase fails, revise the affected script before proceeding — do not output a script you know fails a check.

---

### SCORING
Report the actual counts from PHASE 4, not an estimate. Use this exact structure inside scfsScore, replacing the bracketed parts with real numbers and a short gaps list (write "none" if a category truly has no gaps — do not omit the field):

Coverage:<implemented>/<found found in Phase 1>, Fidelity:<0-100 based on arg/return mismatches found>, Events:<fired>/<found in Phase 1>, Logic:<0-100, reduced for every stub or resource-flow break found>, Security:<0-100>, Overall:<0-100>/100, Gaps:<comma-separated list of specific unfired events, stubbed handlers, or resource-flow breaks found, or "none">

A script that has any unfired event, any stubbed handler, or any resource-flow break listed in Gaps cannot report Overall above 90 — the Gaps list and the Overall number must be consistent with each other. Do not report a perfect score alongside a non-empty Gaps list.

---

### INPUTS

Client script(s) to analyze:
{first}

Additional information:
{second}

Pre-existing server scripts (reuse, don't duplicate):
{third}
"""

BASEREVIEWER_PROMPT = """
You are a senior Roblox Luau engineer performing an independent accuracy review. You are given the original decompiled client script(s) and a server-side implementation that was already generated to fulfill them. Your job is NOT to rewrite the code from scratch or restyle it -- your job is to find where the generated server diverges from what the client actually requires, and fix ONLY those divergences.

Do not trust the generated script's own comments, its scfsScore claims, or the assumption that it is complete. Treat it as a first draft that may contain missing logic, unfired events, stubbed handlers, resource-flow bugs, or fidelity mismatches. Re-derive the truth from the client script yourself before judging the server script against it.

---

### PHASE 1 -- INDEPENDENT CONTRACT RE-EXTRACTION (internal; do this before looking at the generated output in detail)

From the client script(s) alone, build your own table of every remote: path, kind (Event/Function), direction, arguments (name inferred from usage, type inferred from usage, order), return shape for Functions, and the trigger condition for every OnClientEvent listener. Do this exhaustively -- walk every function body, every conditional branch, every loop, every callback passed to task.spawn/coroutine.wrap/pcall. Do not skip a call because it is nested.

For each remote, also note: does the client unpack a single return value, a tuple, or index into a table? Does the same remote get called from more than one place with different argument counts (if so, the union is the real contract)?

Only after this table is built should you open the generated server script and start comparing.

---

### PHASE 2 -- DIFF AGAINST THE GENERATED OUTPUT

Compare your independently-derived contract against the generated server script and identify every divergence in these categories:

1. MISSING REMOTES -- a remote the client calls that has no handler at all in the generated output.
2. UNFIRED EVENTS -- an OnClientEvent listener the client is shown to depend on, where no FireClient/FireAllClients call in the generated output actually triggers it, or triggers it with the wrong arguments.
3. STUBS -- a handler that exists and is connected, but whose body is empty, is only a comment, returns early with no state change, or otherwise does not perform the action the client expects. A handler that "exists" is not the same as a handler that "works."
4. FIDELITY MISMATCHES -- argument order, argument types, or RemoteFunction return shape that does not match how the client actually sends or consumes them.
5. RESOURCE FLOW BREAKS -- this is the category most likely to be silently wrong. For every currency, inventory item, or claimable reward in the script, trace ALL the places that write to it and ALL the places that read from it to pay it out (balance displays, claim remotes, withdraw remotes). A resource is only correctly implemented if every branch that should deliver it to a player (online recipient, offline recipient, gifted, purchased, claimed) writes to the SAME field that the payout/claim path actually reads. A common bug is crediting a "raised" or "received" tracking field on one branch while the real claim remote only ever drains a differently-named field -- meaning the resource is tracked but never actually payable. Trace every currency/item field by name across the entire script before concluding it is correct.
6. SECURITY GAPS -- missing validation, a single cooldown shared across unrelated action types (which lets a high-frequency remote starve a low-frequency one), trusting a client-supplied Instance reference directly, or missing permission checks.
7. DATA CONSISTENCY -- DataStore calls not pcall-wrapped with retry, a claim/consume flow that wipes more state than the specific entries being claimed, or any path that could leave a player's cached data nil.

Do NOT flag something as a bug just because it looks incomplete if the client script genuinely gives no evidence that remote/event/flow should exist -- inventing new scope is itself a mistake. Only flag real divergences from what the client script demonstrates.

---

### PHASE 3 -- FIX ONLY WHAT YOU FOUND

Apply fixes for every divergence found in Phase 2, directly in the server script(s). Preserve everything that is already correct -- do not restyle, rename variables, reorganize sections, or "improve" code that already matches the contract. Minimal, targeted changes only. If pre-existing/reference server scripts were provided below, reuse their naming conventions, remotes folder structure, and DataService interface rather than introducing a new pattern.

If a fix requires assuming something the client script does not make explicit (e.g. a cooldown duration, a page size), pick the most conservative reasonable value and leave a short comment noting it was not directly derivable.

---

### OUTPUT FORMAT (STRICT -- matches the generation format so both stages can be parsed the same way)

Output only raw [SCRIPT_START]...[SCRIPT_END] blocks. No markdown, no prose before/after, no code fences. Only output a block for a script you actually changed -- do NOT re-emit a script you made zero changes to.

[SCRIPT_START]
scriptType = Script | LocalScript | ModuleScript
scriptParent = Full explicit path, unchanged from the original unless the fix required relocating it
scriptParentType = Service | Folder | ModuleScript | Script
scriptName = Same name as the original script being fixed
changesApplied = One line per fix, semicolon-separated, in the form [category] what was wrong (client evidence) -> what was changed. Use category names from Phase 2 (MISSING REMOTE, UNFIRED EVENT, STUB, FIDELITY, RESOURCE FLOW, SECURITY, DATA CONSISTENCY). If this script needed no changes, do not emit a block for it at all.
scfsScore = Coverage:<n>/<n found>, Fidelity:<0-100>, Events:<n fired>/<n found>, Logic:<0-100>, Security:<0-100>, Overall:<0-100>/100, Gaps:<comma-separated remaining concerns, or "none">
scriptContent = 
-- Full corrected Luau source for this script, not just the changed lines.
-- Preserve existing section comments and structure; only alter what was wrong.
[SCRIPT_END]

The Overall score in scfsScore must be consistent with changesApplied and Gaps -- do not report a high Overall score if changesApplied lists a RESOURCE FLOW or STUB fix, and do not report Gaps as "none" unless you actually re-verified all seven Phase 2 categories against your Phase 1 contract table.

If, after full review, no script required any changes, output exactly:
No Changes Required. [Reviewed: <comma-separated script names>]

---

### INPUTS

Client script(s) (the ground truth -- re-derive the contract from these yourself, do not trust the generated output's implied contract):
{client_scripts}

Generated server script(s) to review and fix:
{server_output}

Pre-existing / reference server scripts (match their conventions, do not duplicate their remotes or DataService logic):
{pre_existing_scripts}
"""

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SERVER_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
REVIEW_MODEL = "gemini-3.5-flash"  # fixed model for review

logging.basicConfig(level=logging.INFO)

SCRIPT_BLOCK_RE = re.compile(r"\[SCRIPT_START\](.*?)\[SCRIPT_END\]", re.DOTALL)
SCRIPT_NAME_RE = re.compile(r"scriptName\s*=\s*(.+)")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_model_output(text):
    """Strip common wrapping reasoning/generation models add despite
    'no code fences / no prose' instructions, and drop any preamble
    before the first recognized output marker."""
    if not text:
        return text
    text = text.strip()
    text = re.sub(r"^```(?:lua)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    start_idx = text.find("[SCRIPT_START]")
    no_changes_idx = text.find("No Changes Required")
    no_server_idx = text.find("No Server Script")

    candidates = [i for i in (start_idx, no_changes_idx, no_server_idx) if i > 0]
    if candidates:
        text = text[min(candidates):]
    return text.strip()


def parse_script_blocks(text):
    """Returns dict of scriptName -> full raw block text (including
    the SCRIPT_START/SCRIPT_END markers)."""
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
    """Overlay the reviewer's corrected blocks onto the base generation
    output, keyed by scriptName. Scripts the reviewer didn't touch are
    kept exactly as the generator produced them.

    Returns (merged_text, list_of_changed_script_names).
    """
    review_text = clean_model_output(review_text)

    if not review_text or review_text.startswith("No Changes Required"):
        return base_text, []

    base_blocks = parse_script_blocks(base_text)
    review_blocks = parse_script_blocks(review_text)

    if not review_blocks:
        # Reviewer didn't return anything parseable; don't silently drop it.
        app.logger.warning("Review output not in [SCRIPT_START] format; keeping base output only")
        return base_text, []

    if not base_blocks:
        # Base generation wasn't in the expected block format; can't merge by
        # name, so just append the review output after the base output.
        app.logger.warning("Base output not in [SCRIPT_START] format; skipping merge, returning both raw")
        return base_text + "\n\n" + review_text, list(review_blocks.keys())

    merged = dict(base_blocks)
    changed_names = list(review_blocks.keys())
    merged.update(review_blocks)  # reviewer's fixed versions win, by scriptName

    return "\n\n".join(merged.values()), changed_names


def extract_gemini_output(result):
    """Extract final text from the Google response. Handles the
    'interactions'-style shape this code targets, and falls back to the
    standard generateContent shape in case that isn't actually what's
    being hit by INTERACTIONS_URL."""
    for step in result.get("steps", []):
        if step.get("type") == "model_output":
            content = step.get("content", [])
            texts = [item.get("text", "") for item in content if item.get("type") == "text"]
            if texts:
                return " ".join(texts)

    if "output_text" in result:
        return result["output_text"]
    if "output" in result:
        return result["output"]

    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        pass

    return str(result)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

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

        # --- Generation step ---
        payload = {
            "model": client_ai_model,
            "input": FormattedPrompt,
            "environment": "remote",
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key_to_use,
        }

        app.logger.info(f"Sending generation request to Google with agent: {client_ai_model}")
        resp = requests.post(INTERACTIONS_URL, json=payload, headers=headers, timeout=(10, 300))
        app.logger.info(f"Google responded with status: {resp.status_code}")

        if resp.status_code != 200:
            try:
                error_json = resp.json()
                error_msg = error_json.get("error", {}).get("message", "Unknown error from Google")
                error_details = json.dumps(error_json)
            except Exception:
                error_msg = resp.text[:200]
                error_details = resp.text
            app.logger.error(f"Google API error: {error_msg}")
            return jsonify({"error": "Google API error", "details": error_msg, "raw": error_details}), 500

        result = resp.json()
        output_text = clean_model_output(extract_gemini_output(result))

        if output_text is None or not output_text.strip():
            return jsonify({"error": "Empty output from generation model"}), 500

        # --- Review stage (now always runs, using Gemini 2.5 Flash via the same Google API) ---
        review_result = None
        review_reasoning = None
        final_output = output_text
        changed_scripts = []

        try:
            # Build review prompt
            FormattedReviewPrompt = BASEREVIEWER_PROMPT.format(
                client_scripts=clientscript,
                server_output=output_text,
                pre_existing_scripts=serverside or "None provided.",
            )

            review_payload = {
                "model": REVIEW_MODEL,
                "input": FormattedReviewPrompt,
                "environment": "remote",
            }
            # Use the same headers (same API key)
            review_resp = requests.post(INTERACTIONS_URL, json=review_payload, headers=headers, timeout=(10, 300))
            app.logger.info(f"Review request status: {review_resp.status_code}")

            if review_resp.status_code == 200:
                review_result_json = review_resp.json()
                review_message = extract_gemini_output(review_result_json)
                review_result = clean_model_output(review_message)

                # Merge reviewed blocks into the original generation
                final_output, changed_scripts = merge_reviewed_output(output_text, review_result)
            else:
                # If review fails, log and keep the generation output
                app.logger.error(f"Review API call failed with status {review_resp.status_code}: {review_resp.text[:200]}")
                final_output = output_text

        except Exception as e:
            # A failed review should not lose an already-successful generation.
            app.logger.error(f"Review stage failed, returning unreviewed output: {e}")
            final_output = output_text

        return jsonify({
            "response": final_output,
            "generation_raw": output_text,
            "review_raw": review_result,
            "review_reasoning": review_reasoning,
            "scripts_changed_by_review": changed_scripts,
        })

    except requests.exceptions.RequestException as e:
        app.logger.error(f"RequestException: {e}")
        return jsonify({"error": "Network/Request error", "details": str(e)}), 500
    except KeyError as e:
        app.logger.error(f"KeyError: {e} - response structure unexpected")
        return jsonify({"error": "Unexpected response structure from Google", "details": str(e)}), 500
    except Exception as e:
        app.logger.error(f"Unhandled exception: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
