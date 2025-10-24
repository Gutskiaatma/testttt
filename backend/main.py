from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.llms.together import Together  # Removed usage in code
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI  # For LangChain 1.0.2
import os
import sqlite3
from dotenv import load_dotenv
import re

load_dotenv()

app = Flask(__name__)
CORS(app)

# -----------------------
# Initialize chat models
# -----------------------
chat_flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY", "your_google_key_here"),
    temperature=0.7
)

chat_pro = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY", "your_google_key_here"),
    temperature=0.7
)

chat_openai = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY", "your_openai_key_here"),
    temperature=0.7
)

# -----------------------
# Utility functions
# -----------------------
def clean_markdown(text: str) -> str:
    return re.sub(r'\*{1,3}', '', text)

def init_db():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT,
            question TEXT,
            answer TEXT
        )
    ''')
    conn.commit()
    conn.close()

# -----------------------
# Routes
# -----------------------
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        user_input = data.get("text", "").strip()
        session_name = data.get("session", "default")

        if not user_input:
            return jsonify({"reply": "Please enter a message."}), 400

        conn = sqlite3.connect("chat_history.db", timeout=10)
        cursor = conn.cursor()

        # Check cache
        cursor.execute(
            "SELECT answer FROM chats WHERE question = ? AND session_name = ?",
            (user_input, session_name)
        )
        row = cursor.fetchone()
        if row:
            conn.close()
            return jsonify({"reply": row[0]})

        reply = None
        used_model = "none"

        # -----------------------
        # Fallback logic
        # -----------------------
        try:
            print("👉 Trying Gemini Flash")
            response = chat_flash.invoke([{"role": "user", "content": user_input}])
            reply = response.content if hasattr(response, "content") else str(response)
            used_model = "gemini-flash"
        except Exception as e_flash:
            print("❌ Gemini Flash failed:", e_flash)
            try:
                print("🔁 Trying Gemini Pro")
                response = chat_pro.invoke([{"role": "user", "content": user_input}])
                reply = response.content if hasattr(response, "content") else str(response)
                used_model = "gemini-pro"
            except Exception as e_pro:
                print("❌ Gemini Pro failed:", e_pro)
                try:
                    print("🔁 Trying OpenAI GPT-3.5")
                    response = chat_openai.invoke([
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_input}
                    ])
                    reply = response.content if hasattr(response, "content") else str(response)
                    used_model = "openai-gpt3.5"
                except Exception as e_openai:
                    print("💥 All models failed:", e_openai)
                    return jsonify({
                        "reply": "🚫 All AI services are currently unavailable. Try again later."
                    }), 500

        # -----------------------
        # Preserve full formatting
        # -----------------------
        if reply:
            # Remove extra leading/trailing spaces but keep line breaks
            reply = "\n\n".join([line.strip() for line in reply.splitlines() if line.strip()])

        formatted_reply = f"[{used_model}] {reply}"

        # Save to DB
        cursor.execute(
            "INSERT INTO chats (session_name, question, answer) VALUES (?, ?, ?)",
            (session_name, user_input, formatted_reply)
        )
        conn.commit()
        conn.close()

        return jsonify({"reply": formatted_reply})

    except Exception as e:
        print("🔥 Final Error:", e)
        return jsonify({"reply": f"Unhandled Error: {e}"}), 500


@app.route("/sessions", methods=["GET"])
def list_sessions():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT session_name FROM chats")
    sessions = [row[0] for row in cursor.fetchall()]
    conn.close()
    return jsonify({"sessions": sessions})

@app.route("/history/<session_name>", methods=["GET"])
def session_history(session_name):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT question, answer FROM chats WHERE session_name = ?",
        (session_name,)
    )
    rows = cursor.fetchall()
    conn.close()

    history = []
    for q, a in rows:
        history.append({"sender": "user", "text": q})
        history.append({"sender": "bot", "text": a})

    return jsonify({"history": history})

if __name__ == "__main__":
    init_db()
    app.run(port=5000, debug=True)
