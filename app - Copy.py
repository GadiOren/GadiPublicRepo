import sys
sys.setrecursionlimit(10000)

from flask import Flask, request, jsonify, render_template
import uuid
import os
import threading
from dotenv import load_dotenv
from process_audio import AudioProcessor

app = Flask(__name__)

# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Load your .env replacements file
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ # טעינת משתני סביבה")
load_dotenv("replacements.txt")
print("UPLOAD_FOLDER:", os.getenv("UPLOAD_FOLDER"))
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ # סוף טעינת משתני סביבה")

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ # UPLOAD_FOLDER  " + UPLOAD_FOLDER)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

processing_status = {}

# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def preload_heavy_models():
    """
    טוען מראש את המודולים הכבדים (WhisperX, Alignment ומודל זיהוי דוברים)
    """
    print("🔄 טעינת מודולים כבדים (WhisperX, Alignment, וזיהוי דוברים) בזמן אתחול האפליקציה…")
    try:
        from TranscribeAI import lazy_load_models
        lazy_load_models()
        print("✅ טעינת המודולים הכבדים הושלמה בהצלחה.")
    except Exception as e:
        print("❌ שגיאה בטעינת המודולים הכבדים:", e)

models_loaded = False

def load_models_in_background():
    global models_loaded
    if not models_loaded:
        preload_heavy_models()
        models_loaded = True

# ───────────  **NEW**: preload once at import time  ───────────
print("🔧 טוען מודלים כבדים (lazy_load_models)…")  # ← THIS WILL NOW APPEAR IN THE LOGS
load_models_in_background()
# ────────────────────────────────────────────────────────────────

@app.route("/")
def serve_index():
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "לא נבחר קובץ"}), 400

    file = request.files["file"]
    processing_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, f"{processing_id}.wav")
    file.save(file_path)

    processing_status[processing_id] = {"status": "🔄 התחלת עיבוד…"}
    print(f"📌 מזהה עיבוד שנוצר: {processing_id}")

    processor = AudioProcessor(processing_id, file_path, processing_status)
    threading.Thread(target=processor.process).start()

    return jsonify({"processingId": processing_id})

@app.route("/api/status/<processing_id>", methods=["GET"])
def get_status(processing_id):
    print(f"📌 בדיקת סטטוס למזהה: {processing_id}")
    return jsonify(processing_status.get(processing_id, {"status": "❌ לא נמצא"}))

if __name__ == "__main__":
    print("Running app.py (standalone)")
    # For direct 'python app.py' runs, you can still use a background thread,
    # but it's not needed under Gunicorn with preload_app=True.
    threading.Thread(target=load_models_in_background, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
