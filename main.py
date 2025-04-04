from fastapi import FastAPI, UploadFile, File, BackgroundTasks, WebSocket, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid
import asyncio

# ייבוא הפונקציות מהקובץ TranscribeAI.py (ודא ששמות הפונקציות תואמים)
from TranscribeAI import process_audio_file, merge_word_and_segment_data

# יצירת האפליקציה
app = FastAPI(
    title="Transcribe API",
    description="שירות לעיבוד קבצי אודיו עם WhisperX וזיהוי דוברים",
    version="1.0"
)

# הגדרת נתיב לתבניות HTML (templates)
templates = Jinja2Templates(directory="templates")

# הגדרת mount לקבצים סטטיים (למשל, favicon)
app.mount("/static", StaticFiles(directory="static"), name="static")

# מילון גלובלי לניהול סטטוס ותוצאות העבודה (job_id => {status, result})
job_results = {}

# יצירת תיקייה זמנית לאחסון קבצים
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


# Endpoint לטיפול בבקשות ל-favicon – מפנה את הבקשה לקובץ ה-ICO שב-static
@app.get("/favicon.ico")
async def favicon():
    return RedirectResponse(url="/static/Transcription_AI_Gadi_Oren.ico")


# הגשת דף HTML ראשי (ממשק משתמש) בכתובת /
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Endpoint לקבלת קובץ אודיו והפעלת העיבוד ברקע
@app.post("/process-audio")
async def process_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_DIR, f"{job_id}_{file.filename}")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # אתחול סטטוס העבודה במילון
    job_results[job_id] = {"status": "processing", "result": None}
    background_tasks.add_task(process_and_update, job_id, file_path)
    return {"job_id": job_id, "message": "קובץ התקבל, העיבוד מתבצע ברקע."}


# פונקציה שמריצה את העיבוד ומעדכנת את סטטוס העבודה והתוצאה
def process_and_update(job_id: str, file_path: str):
    try:
        df_words, aligned_result = process_audio_file(file_path)
        df_sentences = merge_word_and_segment_data(aligned_result)
        # שמירת קבצי CSV (לא בוטלה)
        df_words.to_csv("full_transcription_words.csv", index=False, encoding="utf-8-sig")
        df_sentences.to_csv("full_transcription_sentences.csv", index=False, encoding="utf-8-sig")

        # הכנת התוצאה כ־JSON (מומרת למילון)
        result_data = {
            "words": df_words.to_dict(orient="records"),
            "sentences": df_sentences.to_dict(orient="records")
        }
        job_results[job_id] = {"status": "completed", "result": result_data}
    except Exception as e:
        print("שגיאה בתהליך העיבוד:", e)
        job_results[job_id] = {"status": "failed", "error": str(e)}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# Endpoint לבדיקה של סטטוס העבודה דרך מזהה העבודה (job_id)
@app.get("/job-status/{job_id}")
async def job_status(job_id: str):
    if job_id in job_results:
        return job_results[job_id]
    return {"error": "מזהה עבודה לא קיים"}


# WebSocket למעקב בזמן אמת אחר סטטוס העבודה
@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    while True:
        if job_id in job_results:
            job_data = job_results[job_id]
            # שליחת כל נתוני העבודה (סטטוס ותוצאה) ללקוח
            await websocket.send_json(job_data)
            if job_data["status"] in ["completed", "failed"]:
                break
        await asyncio.sleep(2)  # בדיקה כל 2 שניות
    await websocket.close()


# הרצת השרת אם הקובץ מופעל כ-main
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
