from flask import Flask, request, jsonify, render_template
import uuid
import os
import threading
from dotenv import load_dotenv
from process_audio import AudioProcessor

app = Flask(__name__)

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ # טעינת משתני סביבה")
# טעינת משתני סביבה
#load_dotenv()
load_dotenv("replacements.txt")
print("UPLOAD_FOLDER:", os.getenv("UPLOAD_FOLDER"))

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ # סוף טעינת משתני סביבה")

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
#os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ # UPLOAD_FOLDER  " + UPLOAD_FOLDER)


# מילון לניהול סטטוס העיבוד
processing_status = {}

@app.route("/")
def serve_index():
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def upload_file():
    """ מקבל קובץ שמע, שומר אותו, ומתחיל עיבוד ברקע """
    if "file" not in request.files:
        return jsonify({"error": "לא נבחר קובץ"}), 400

    file = request.files["file"]
    processing_id = str(uuid.uuid4())  # מזהה ייחודי
    file_path = os.path.join(UPLOAD_FOLDER, f"{processing_id}.wav")
    file.save(file_path)

    # אתחול הסטטוס
    processing_status[processing_id] = {"status": "🔄 התחלת עיבוד..."}
    print(f"📌 מזהה עיבוד שנוצר: {processing_id}")

    # יצירת מופע של מעבד השמע והפעלתו ב-Thread
    processor = AudioProcessor(processing_id, file_path, processing_status)
    threading.Thread(target=processor.process).start()

    return jsonify({"processingId": processing_id})

@app.route("/api/status/<processing_id>", methods=["GET"])
def get_status(processing_id):
    """ מחזיר את הסטטוס והתוצאה אם העיבוד הסתיים """
    print(f"📌 בדיקת סטטוס למזהה: {processing_id}")
    if processing_id in processing_status:
        return jsonify(processing_status[processing_id])
    return jsonify({"status": "❌ לא נמצא"})

if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=5000, debug=True)
    print("Running app.py")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
