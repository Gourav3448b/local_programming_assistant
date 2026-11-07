"""Local Flask web interface for the from-scratch MiniGPT programming tutor."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, render_template, request

from local_llm.assistant import ProgrammingAssistant

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "programming_qa.jsonl"
CHECKPOINT_PATH = BASE_DIR / "artifacts" / "minigpt.pt"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024  # Keep local requests small.

_assistant: ProgrammingAssistant | None = None
_assistant_lock = Lock()


def get_assistant() -> ProgrammingAssistant:
    global _assistant
    if _assistant is None:
        with _assistant_lock:
            if _assistant is None:
                _assistant = ProgrammingAssistant(DATASET_PATH, CHECKPOINT_PATH)
    return _assistant


@app.get("/")
def index():
    assistant = get_assistant()
    return render_template("index.html", model_loaded=assistant.model_loaded)


@app.get("/api/status")
def status():
    assistant = get_assistant()
    return jsonify({
        "model_loaded": assistant.model_loaded,
        "checkpoint": CHECKPOINT_PATH.name if assistant.model_loaded else None,
        "device": str(assistant.device),
        "dataset_examples": len(assistant.examples),
        "model_info": assistant.checkpoint_info,
    })


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    question = payload.get("message", "")
    if not isinstance(question, str):
        return jsonify({"error": "Message must be text."}), 400
    question = question.strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400
    if len(question) > 2000:
        return jsonify({"error": "Please keep the question under 2,000 characters."}), 413
    try:
        result = get_assistant().answer(question)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # Return a safe error; details stay in the server console.
        app.logger.exception("Assistant request failed")
        return jsonify({"error": "The assistant encountered an error. Check the terminal for details."}), 500


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Request is too large."}), 413


if __name__ == "__main__":
    print("Starting the local programming assistant at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop. The app does not call external AI APIs.")
    app.run(host="0.0.0.0", port=5000, debug=False)
