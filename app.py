# app.py
import os
import re
import json
import time
import traceback
import difflib
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, session
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
import markdown
import google.generativeai as genai  # Gemini SDK

# -------------------------
# Config & env
# -------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Track quota usage manually
_current_key_idx = 0
API_USAGE = [0] * len(GEMINI_API_KEY)   # one counter per key
API_LIMIT = 50  # set your quota limit

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env file")

genai.configure(api_key=GEMINI_API_KEY)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OVERRIDES_FILE = os.path.join(APP_DIR, "overrides.json")
FUZZY_CUTOFF = 0.85  # 0.0-1.0: how close questions must be to use an override

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")

# in-memory caches
CONVERSATIONS = {}   # session_id -> history
_OVERRIDES = {}      # loaded from disk: norm_question -> override dict


# -------------------------
# Overrides persistence
# -------------------------
def load_overrides():
    global _OVERRIDES
    try:
        if os.path.exists(OVERRIDES_FILE):
            with open(OVERRIDES_FILE, "r", encoding="utf-8") as f:
                _OVERRIDES = json.load(f)
        else:
            _OVERRIDES = {}
    except Exception:
        traceback.print_exc()
        _OVERRIDES = {}

def save_overrides():
    try:
        with open(OVERRIDES_FILE, "w", encoding="utf-8") as f:
            json.dump(_OVERRIDES, f, ensure_ascii=False, indent=2)
    except Exception:
        traceback.print_exc()


# load at startup
load_overrides()


# -------------------------
# Helpers
# -------------------------
def get_sid():
    if "sid" not in session:
        session["sid"] = os.urandom(8).hex()
    return session["sid"]

_norm_regex = re.compile(r"[^\w\s]")  # remove punctuation


def normalize_question(q: str) -> str:
    if not q:
        return ""
    s = q.lower()
    s = _norm_regex.sub("", s)           # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()   # normalize whitespace
    return s


def google_search_snippets(query, num=5):
    # simple scraping of Google search snippets (best-effort)
    import requests
    url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        snippets = []
        seen = set()
        for g in soup.select("div.BNeawe.s3v9rd.AP7Wnd"):
            txt = g.get_text(" ", strip=True)
            if txt and txt not in seen:
                snippets.append({"text": txt, "url": url})
                seen.add(txt)
            if len(snippets) >= num:
                break
        return snippets
    except Exception:
        return []


def call_gemini(prompt, model_name="gemini-1.5-pro-latest", depth=0):
    global API_USAGE, _current_key_idx
    try:
        # Count usage for current key
        API_USAGE[_current_key_idx] += 1

        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        if resp and getattr(resp, "candidates", None):
            return resp.text.strip()
        return "[Gemini Error: Empty response]"
    except Exception as e:
        app._last_error = str(e)   # store last error for /_status
        if model_name == "gemini-1.5-pro-latest" and depth < 1:
            time.sleep(1)
            return call_gemini(prompt, "gemini-1.5-flash-latest", depth + 1)
        return f"[Gemini Error: {str(e)}]"

# -------------------------
# Prompts (concise output)
# -------------------------
BASE_PROMPT = """You are an Aptitude Problem Solver bot.
Answer concisely and only in this structure:

1) Formula used: <one line>
2) Process: <short steps, max 3 lines>
3) Final Answer: ✅ <final value and option (if any)>

Question: {question}
Reference Snippets:
{snippets}
"""

MODIFY_PROMPT = """You are revising a previous aptitude solution.

Previous solution:
{prev_solution}

User feedback:
{feedback}

Provide corrected output exactly in the same short format:
1) Formula used: <one line>
2) Process: <short steps, max 3 lines>
3) Final Answer: ✅ <final value and option (if any)>

Question: {question}
Reference Snippets:
{snippets}
"""


# -------------------------
# Ask route (checks overrides first)
# -------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/ask", methods=["POST"])
def ask():
    try:
        sid = get_sid()
        q = request.form.get("question", "").strip()
        image_file = request.files.get("image")

        if not q and image_file:
            img = Image.open(image_file.stream)
            q = pytesseract.image_to_string(img).strip()

        if not q:
            return jsonify({"error": "No question provided"}), 400

        norm = normalize_question(q)

        # 1) exact override?
        override = _OVERRIDES.get(norm)

        # 2) fuzzy match override?
        used_override_note = None
        if not override:
            keys = list(_OVERRIDES.keys())
            if keys:
                matches = difflib.get_close_matches(norm, keys, n=1, cutoff=FUZZY_CUTOFF)
                if matches:
                    override = _OVERRIDES[matches[0]]
                    used_override_note = f"Note: using corrected answer for a similar question (match: {matches[0]})"

        # If override exists, return stored corrected answer (no Gemini call)
        if override:
            reply_html = override.get("html") or markdown.markdown(override.get("raw", ""), extensions=["fenced_code", "tables"])
            # add small note banner if fuzzy matched or override
            if used_override_note:
                reply_html = f"<div style='font-size:0.9em;color:#555;margin-bottom:8px'><em>{used_override_note}</em></div>" + reply_html
            # save to current session history
            if sid not in CONVERSATIONS:
                CONVERSATIONS[sid] = {"history": []}
            CONVERSATIONS[sid]["history"].append({"q": q, "a": override.get("raw"), "override_used": True})
            return jsonify({"reply": reply_html})

        # No override → fetch snippets + ask Gemini
        snippets = google_search_snippets(q)
        snippets_str = "\n\n".join([f"{s['url']}:\n{s['text']}" for s in snippets]) or "None"

        prompt = BASE_PROMPT.format(question=q, snippets=snippets_str)
        reply_raw = call_gemini(prompt)
        reply_html = markdown.markdown(reply_raw, extensions=["fenced_code", "tables"])

        # store in session history
        if sid not in CONVERSATIONS:
            CONVERSATIONS[sid] = {"history": []}
        CONVERSATIONS[sid]["history"].append({"q": q, "a": reply_raw, "snippets": snippets})

        return jsonify({"reply": reply_html})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/_status", methods=["GET"])
def status():
    return jsonify({
        "current_key_index": _current_key_idx,
        "total_keys": len(GEMINI_API_KEY),
        "current_key": GEMINI_API_KEY[_current_key_idx][:8] + "****",
        "usage": API_USAGE[_current_key_idx],
        "limit": API_LIMIT,
        "last_error": getattr(app, "_last_error", None)
    })



# -------------------------
# Modify route (saves override)
# -------------------------
@app.route("/modify", methods=["POST"])
def modify():
    """
    Expects assistant_text (optional) and feedback (optional).
    Finds the last conversation entry (or matching assistant_text) and asks Gemini to revise.
    Then saves the revision to overrides.json (so future users get it).
    """
    try:
        sid = get_sid()
        data = CONVERSATIONS.get(sid)
        if not data or not data.get("history"):
            return jsonify({"error": "No history available to modify."}), 400

        assistant_text = request.form.get("assistant_text", "").strip()
        feedback = request.form.get("feedback", "User says it's wrong, please fix.").strip()

        # find target history entry
        target_idx = None
        if assistant_text:
            for i in range(len(data["history"]) - 1, -1, -1):
                if data["history"][i].get("a", "").strip() == assistant_text:
                    target_idx = i
                    break
        if target_idx is None:
            target_idx = len(data["history"]) - 1

        entry = data["history"][target_idx]
        question = entry.get("q")
        prev_solution = entry.get("a", "")

        # refresh snippets for context (best-effort)
        snippets = entry.get("snippets", []) or google_search_snippets(question)
        snippets_str = "\n\n".join([f"{s['url']}:\n{s['text']}" for s in snippets]) or "None"

        prompt = MODIFY_PROMPT.format(
            prev_solution=prev_solution,
            feedback=feedback,
            question=question,
            snippets=snippets_str
        )

        # get revised answer from Gemini
        revised_raw = call_gemini(prompt)
        revised_html = markdown.markdown(revised_raw, extensions=["fenced_code", "tables"])

        # Save override (normalized question -> revision)
        norm = normalize_question(question)
        override_entry = {
            "raw": revised_raw,
            "html": revised_html,
            "modified_by": sid,
            "modified_at": datetime.utcnow().isoformat() + "Z",
            "feedback": feedback
        }
        _OVERRIDES[norm] = override_entry
        save_overrides()

        # Append modified to history
        data["history"].append({
            "q": question,
            "a": revised_raw,
            "modified": True,
            "feedback": feedback
        })

        # Return revised HTML
        note = "<div style='font-size:0.9em;color:#555;margin-bottom:8px'><em>Saved corrected answer and will be served to future matching questions.</em></div>"
        return jsonify({"reply": note + revised_html})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# -------------------------
# Admin: list overrides (optional)
# -------------------------
@app.route("/_overrides", methods=["GET"])
def list_overrides():
    # returns a small debug view of overrides (for dev)
    keys = [{"question": k, "modified_at": _OVERRIDES[k].get("modified_at")} for k in _OVERRIDES]
    return jsonify({"count": len(keys), "overrides": keys})

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
 