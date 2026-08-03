from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid

from agents import process_input

app = Flask(__name__)
CORS(app)

# Temporary session storage
# Later you can replace this with SQLite
sessions = {}


@app.route("/start", methods=["GET"])
def home():

    return jsonify({
        "message": "Santhosh Portfolio Assistant API is running."
    })


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    if data is None:
        return jsonify({
            "status": "error",
            "message": "Request body is missing."
        }), 400

    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "status": "error",
            "message": "Question is required."
        }), 400

    session_id = data.get("session_id")

    if not session_id:
        session_id = str(uuid.uuid4())

    result = process_input(question)

    sessions.setdefault(session_id, []).append({
        "question": question,
        "answer": result["final_output"]
    })

    return jsonify({

        "status": "success",

        "session_id": session_id,

        "question": question,

        "answer": result["final_output"],

        "sources": result["sources"]

    })


@app.route("/history/<session_id>", methods=["GET"])
def history(session_id):

    return jsonify({

        "session_id": session_id,

        "history": sessions.get(session_id, [])

    })


@app.route("/clear/<session_id>", methods=["DELETE"])
def clear(session_id):

    sessions.pop(session_id, None)

    return jsonify({

        "status": "success",

        "message": "Session deleted."

    })


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)