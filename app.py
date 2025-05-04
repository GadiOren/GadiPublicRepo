import os
import sys
sys.setrecursionlimit(10000)

# ─── 1. Load local .env / replacements.txt (if present) ─────────────────────────
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv("replacements.txt", usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path)
    print("🔸 replacements.txt נטען")
else:
    print("🔸 לא נמצא replacements.txt – סומכים על ENV בפלטפורמה")

# ─── 2. If running in GAE-Flex, pull secrets from Secret Manager ───────────────
if os.getenv("GAE_ENV", "").startswith("flex") or os.getenv("K_SERVICE"):
    print("🔐 טוען סודות מ-Secret Manager…")
    try:
        from google.cloud import secretmanager
        sm = secretmanager.SecretManagerServiceClient()
        project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
        def _get_secret(secret_id: str) -> str:
            name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
            r = sm.access_secret_version(request={"name": name})
            return r.payload.data.decode("UTF-8")
        os.environ["OPENAI_API_KEY"]    = _get_secret("OPENAI_API_KEY")
        os.environ["HUGGING_FACE_TOKEN"] = _get_secret("HUGGING_FACE_TOKEN")
        print("✅ סודות נטענו בהצלחה.")
    except Exception as e:
        print("❌ שגיאת טעינת סודות:", e)
else:
    print("🏠 ריצה מקומית: משתמש במשתני סביבה קיימים.")

# ─── 3. Import modules that depend on those keys ────────────────────────────────
import uuid
import threading
from flask import Flask, request, jsonify, render_template
from process_audio import AudioProcessor
from TranscribeAI import lazy_load_models

# ─── 4. Create Flask app ───────────────────────────────────────────────────────
app = Flask(__name__)

# ─── 5. Eager-in-background: start loading heavy models immediately ─────────────
_models_loaded = False
def _load_models_bg():
    print("🔧 (background) טוען מודולים כבדים…", flush=True)
    try:
        lazy_load_models()
        print("✅ מודלים נטענו בהצלחה ברקע.")
    except Exception as e:
        print("❌ שגיאה בטעינת המודלים ברקע:", e)

def start_model_loader():
    global _models_loaded
    if not _models_loaded:
        threading.Thread(target=_load_models_bg, daemon=True).start()
        _models_loaded = True

# kick off the loader right away
start_model_loader()

# ─── 6. Upload folder & status store ────────────────────────────────────────────
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print("UPLOAD_FOLDER:", UPLOAD_FOLDER)

processing_status = {}

# ─── 7. Health-check endpoint (for readiness checks) ───────────────────────────
@app.route("/")
def health_check():
    return jsonify({"status": "ok"}), 200

# ─── 8. UI endpoint ─────────────────────────────────────────────────────────────
@app.route("/ui")
def serve_index():
    return render_template("index.html")

# ─── 9. Upload endpoint ────────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload_file():
    start_model_loader()

    if "file" not in request.files:
        return jsonify({"error": "לא נבחר קובץ"}), 400

    f = request.files["file"]
    pid = str(uuid.uuid4())
    path = os.path.join(UPLOAD_FOLDER, f"{pid}.wav")
    f.save(path)

    processing_status[pid] = {"status": "🔄 התחלת עיבוד..."}
    print(f"📌 מזהה עיבוד: {pid}")

    proc = AudioProcessor(pid, path, processing_status)
    threading.Thread(target=proc.process, daemon=True).start()

    return jsonify({"processingId": pid})

# ─── 10. Status endpoint ───────────────────────────────────────────────────────
@app.route("/api/status/<pid>", methods=["GET"])
def get_status(pid):
    print(f"📌 סטטוס לבקשה: {pid}")
    return jsonify(processing_status.get(pid, {"status": "❌ לא נמצא"}))

# ─── 11. Run locally ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 מריץ Flask מקומי ב-:8080")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
