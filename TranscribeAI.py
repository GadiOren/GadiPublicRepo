import os
import sys
import subprocess
import torch
import whisperx
import pandas as pd
import math
from chatGpt_Improvement import generate_title_summary_and_speakers, correct_transcription_and_summary
from pyannote.audio import Pipeline

# ─── ensure Hugging Face cache dir env var ────────────────────────────────────────  # ◀ ADDED
HF_CACHE = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))            # ◀ ADDED
os.environ["HF_HOME"] = HF_CACHE

# הגדרות גלובליות
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN") or "YOUR_HF_TOKEN"

# משתנים גלובליים למודלים – יאתחלו במצב None (טעינה דלה)
WHISPER_MODEL = None
ALIGNMENT_MODEL = None
METADATA = None
DIARIZATION_PIPELINE = None


import os
import sys
import subprocess
import torch
import whisperx
import pandas as pd
import math
from chatGpt_Improvement import generate_title_summary_and_speakers, correct_transcription_and_summary
from pyannote.audio import Pipeline

# ─── ensure Hugging Face cache dir env var ────────────────────────────────────────  # ◀ ADDED
HF_CACHE = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))            # ◀ ADDED
os.environ["HF_HOME"] = HF_CACHE                                                        # ◀ ADDED

# הגדרות גלובליות
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN") or "YOUR_HF_TOKEN"

# משתנים גלובליים למודלים – יאתחלו במצב None (טעינה דלה)
WHISPER_MODEL = None
ALIGNMENT_MODEL = None
METADATA = None
DIARIZATION_PIPELINE = None


def lazy_load_models():
    """
    טוען את המודלים הכבדים (WhisperX, Alignment ומודל זיהוי הדוברים)
    במצב lazy – רק כאשר הם נדרשים לעיבוד האודיו.
    """
    global WHISPER_MODEL, ALIGNMENT_MODEL, METADATA, DIARIZATION_PIPELINE

    # ─── ensure the faster‑whisper model is fully in cache ───────────────────────────  # ◀ ADDED
    if WHISPER_MODEL is None:
        print("🔄 pre‑downloading faster‑whisper‑large‑v2 into cache…")                    # ◀ ADDED
        try:
            from huggingface_hub import snapshot_download                                  # ◀ ADDED
            snapshot_download(                                                          # ◀ ADDED
                repo_id="Systran/faster-whisper-large-v2",
                cache_dir=HF_CACHE,
                use_auth_token=HUGGING_FACE_TOKEN
            )                                                                           # ◀ ADDED
        except Exception as e:                                                          # ◀ ADDED
            print(f"⚠️ failed to pre‑download model, will retry at load: {e}")          # ◀ ADDED

    # ═══════════════ actual lazy loading ════════════════════════════════════════
    if WHISPER_MODEL is None:
        print("טעינת מודל WhisperX (lazy)…")
        WHISPER_MODEL = whisperx.load_model("large-v2", device=DEVICE, compute_type="float32")
    if ALIGNMENT_MODEL is None or METADATA is None:
        print("טעינת מודל Alignment (lazy)…")
        ALIGNMENT_MODEL, METADATA = whisperx.load_align_model(language_code="he", device=DEVICE)
    if DIARIZATION_PIPELINE is None:
        print("טעינת מודל זיהוי דוברים (pyannote) (lazy)…")
        try:
            DIARIZATION_PIPELINE = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=HUGGING_FACE_TOKEN
            )
            print("מודל זיהוי דוברים נטען בהצלחה!")
        except Exception as e:
            print("טעינת מודל זיהוי דוברים נכשלה:", e)
            DIARIZATION_PIPELINE = None


# … the rest of the file is unchanged …
def format_time(seconds):
    if seconds is None:
        return "N/A"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02}:{s:02}"

def assign_speakers_to_words(word_segments, speaker_segments):
    for word in word_segments:
        word_start = word.get("start", 0)
        word["speaker"] = "לא ידוע"
        for seg in speaker_segments:
            if seg["start"] <= word_start <= seg["end"]:
                word["speaker"] = seg["speaker"]
                break
    return word_segments




def run_diarization(abs_path):
    """
    מריץ זיהוי דוברים באמצעות המודל שנטען (Lazy) אם הוא עדיין לא נטען.
    מחזירה רשימת סגמנטים עם זמני התחלה, סיום והדובר.
    """
    # טוענים את המודל במידת הצורך
    if DIARIZATION_PIPELINE is None:
        lazy_load_models()
    speaker_segments = []
    if DIARIZATION_PIPELINE is None:
        print("⚠️ מודל זיהוי דוברים לא זמין, ממשיכים ללא זיהוי דוברים.")
        return speaker_segments

    try:
        print("🔹 מבצע זיהוי דוברים...")
        diarization_result = DIARIZATION_PIPELINE({"uri": abs_path, "audio": abs_path})
        print("✅ זיהוי דוברים הצליח! להלן הפלט:")
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            print(f"{turn.start:.2f}s - {turn.end:.2f}s | דובר: {speaker}")
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
    except Exception as e:
        print("❌ שגיאה בעת ביצוע זיהוי דוברים. ממשיכים ללא זיהוי דוברים.")
        print(e)

    return speaker_segments


def process_audio_file(file_path):
    """
    פונקציה מרכזית לעיבוד קובץ שמע:
      1. בודקת את FFmpeg, את קיום קובץ האודיו.
      2. טוענת את המודלים במצב lazy במידת הצורך.
      3. מבצעת תמלול ו-Alignment.
      4. מריצה זיהוי דוברים.
      5. מאחדת נתונים ובונה DataFrame ברמת המילה.
    """
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        print("✅ FFmpeg זמין!")
    except FileNotFoundError:
        print("❌ FFmpeg לא נמצא במערכת או אינו מוגדר ב-PATH.")
        sys.exit(1)

    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        print(f"❌ הקובץ {abs_path} לא קיים!")
        sys.exit(1)
    print(f"🔹 מעבד קובץ אודיו: {abs_path}")

    # טעינה דחויה של המודלים – רק כאשר נדרש
    lazy_load_models()

    print("🔹 מבצע load_audio ל WhisperX...")
    audio = whisperx.load_audio(abs_path)
    print("🔹 מבצע תמלול ראשוני באמצעות WhisperX...")

    whisper_result = WHISPER_MODEL.transcribe(audio)
    print("🔹 מסיים תמלול. שפה:", whisper_result.get("language", "לא ידוע"))

    print("🔹 מבצע Alignment...")
    aligned_result = whisperx.align(
        whisper_result["segments"],
        ALIGNMENT_MODEL,
        METADATA,
        audio,
        DEVICE
    )
    print("✅ Alignment הושלם!")

    speaker_segments = run_diarization(abs_path)

    aligned_words = []
    if "segments" in aligned_result:
        for segment in aligned_result["segments"]:
            if "words" in segment:
                aligned_words.extend(segment["words"])
    else:
        print("❌ לא נמצאו segments בתוצאות Alignment.")

    if speaker_segments:
        aligned_words = assign_speakers_to_words(aligned_words, speaker_segments)

    data = []
    for w in aligned_words:
        word_text = w.get("word", "").strip()
        start_time = w.get("start")
        end_time = w.get("end")
        score = w.get("score")
        speaker = w.get("speaker", "לא ידוע")

        if not word_text or start_time is None or end_time is None:
            continue

        try:
            predict_value = score if isinstance(score, (int, float)) else "N/A"
            score_percent = round(score * 100, 2) if isinstance(score, (int, float)) else "N/A"
        except:
            predict_value = "N/A"
            score_percent = "N/A"

        data.append({
            "מילה": word_text,
            "Predict": predict_value,
            "score": score,
            "אחוז ניבוי": score_percent,
            "התחלה": format_time(start_time),
            "סיום": format_time(end_time),
            "משך": round(end_time - start_time, 2),
            "דובר": speaker
        })

    df_words = pd.DataFrame(data)
    df_words.sort_values(by=["התחלה"], inplace=True)

    df_words["Predict_numeric"] = pd.to_numeric(df_words["Predict"], errors="coerce")
    mean_score = df_words["Predict_numeric"].mean()
    std_score = df_words["Predict_numeric"].std()
    dynamic_threshold = mean_score - std_score
    print(f"ממוצע score: {mean_score:.4f}, סטיית תקן: {std_score:.4f}, סף דינמי: {dynamic_threshold:.4f}")

    df_words["בעייתית"] = df_words["Predict_numeric"].apply(
        lambda x: True if pd.notnull(x) and x < dynamic_threshold else False)
    print("✅ עיבוד ברמת המילה הסתיים בהצלחה!")
    return df_words, aligned_result


def merge_word_and_segment_data(aligned_result):
    """
    ממזג נתוני Alignment כך שכל שורה ב-DataFrame מייצגת משפט/פסקה.
    במידה ואין תוצאות, מחזיר DataFrame ריק.
    """
    sentences = []
    if "segments" in aligned_result and aligned_result["segments"]:
        for seg in aligned_result["segments"]:
            text = seg.get("text", "").strip()
            start = seg.get("start")
            end = seg.get("end")
            if start is None or end is None or not text:
                continue
            duration = end - start

            words = seg.get("words", [])
            if words:
                scores = [w.get("score") for w in words if isinstance(w.get("score"), (int, float))]
                avg_score = sum(scores) / len(scores) if scores else None
                std_seg = math.sqrt(sum((s - avg_score) ** 2 for s in scores) / len(scores)) if scores else 0
                seg_threshold = avg_score - std_seg if avg_score is not None else None

                speakers = [w.get("speaker", "לא ידוע") for w in words]
                majority_speaker = max(set(speakers), key=speakers.count) if speakers else "לא ידוע"

                problematic_words = [w.get("word", "").strip() for w in words
                                     if
                                     isinstance(w.get("score"), (int, float)) and seg_threshold is not None and w.get(
                                         "score") < seg_threshold]
            else:
                avg_score = None
                majority_speaker = "לא ידוע"
                problematic_words = []

            sentences.append({
                "משפט": text,
                "התחלה": format_time(start),
                "סיום": format_time(end),
                "משך": round(duration, 2),
                "ממוצע אחוז ניבוי": round(avg_score * 100, 2) if avg_score is not None else "N/A",
                "דובר": majority_speaker,
                "מילים בעייתיות": problematic_words
            })
    else:
        print("❌ לא נמצאו segments בתוצאות Alignment. מחזיר DataFrame ריק.")
        return pd.DataFrame(columns=["משפט", "התחלה", "סיום", "משך", "ממוצע אחוז ניבוי", "דובר", "מילים בעייתיות"])

    df_sentences = pd.DataFrame(sentences)
    return df_sentences


if __name__ == "__main__":
    # קריאה של קובץ CSV עם עמודות 'משפט' ו-'דובר'
    df = pd.read_csv("full_transcription_sentences.csv")
    transcript_df = df[['משפט', 'דובר']].copy()
    audio_file = "example_audio.wav"

    # הפקת כותרת, תקציר, רשימת דוברים ותמלול מעודכן
    title, summary, speakers, updated_transcript_df = generate_title_summary_and_speakers(transcript_df, audio_file)
    print("\n--- תוצאות הפקת כותרת, תקציר ורשימת דוברים ---")
    print("כותרת:", title)
    print("סיכום:", summary)
    print("רשימת דוברים:", speakers)
    print("\n--- תמלול השיחה המעודכן ---")
    print(updated_transcript_df.head(20).to_string(index=False))

    # תיקון התמלול והפקת הפלט הסופי
    result = correct_transcription_and_summary(updated_transcript_df, title, summary, speakers, audio_file)

    print("\n--- תוצאות תיקון התמלול ---")
    print("שם הקובץ:", audio_file)
    print("כותרת מתוקנת:", result["title"])
    print("סיכום מתוקן:", result["summary"])
    print("רשימת דוברים לאחר תיקון:", result["speakers"])
    print("\n--- תמלול שיחה חדש לאחר תיקון ---")
    print(result["transcript_after"])
    print("\n--- תמלול שיחה לפני תיקון ---")
    print(result["transcript_before"])
