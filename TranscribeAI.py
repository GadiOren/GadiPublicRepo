import os
import sys
import subprocess
import torch
import whisperx
import pandas as pd
import math
from chatGpt_Improvement import generate_title_summary_and_speakers, correct_transcription_and_summary


# (לא חובה – לשם הדפסה מסודרת)
from tabulate import tabulate

# נייבא את Pipeline מ־pyannote.audio לשימוש בזיהוי דוברים (לרמת המשפט)
from pyannote.audio import Pipeline

# --------------------- הגדרות גלובליות ---------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# טוקן ל-Hugging Face נטען ממשתנה הסביבה
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
HUGGING_FACE_TOKEN = "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"

# הגדרת סף בסיסי לזיהוי מילים בעייתיות (לשימוש עזר ברמת המילה)
THRESHOLD_SCORE = 0.05

def format_time(seconds):
    """פורמט שניות ל-MM:SS עבור הצגה ידידותית"""
    if seconds is None:
        return "N/A"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02}:{s:02}"

def assign_speakers_to_words(word_segments, speaker_segments):
    """
    משייך לכל מילה דובר על סמך זמן ההתחלה שלה.
    עבור כל מילה, בודקים אם זמן ההתחלה נופל בתוך סגמנט של דובר.
    """
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
    טוען ומריץ את מודל זיהוי הדוברים באמצעות Pipeline.from_pretrained
    עבור קובץ שמע נתון.
    מחזירה רשימת סגמנטים עם זמני התחלה, סיום והדובר.
    """
    speaker_segments = []
    try:
        print("🔹 טוען מודל זיהוי דוברים (pyannote/speaker-diarization)...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=HUGGING_FACE_TOKEN
        )
        print("✅ מודל זיהוי דוברים נטען בהצלחה!")
    except Exception as e:
        print(f"❌ טעינת מודל זיהוי דוברים נכשלה: {e}")
        print("⚠️ ממשיך ללא זיהוי דוברים...")
        return speaker_segments

    try:
        print("🔹 מבצע זיהוי דוברים...")
        diarization_result = diarization_pipeline({"uri": abs_path, "audio": abs_path})
        print("✅ זיהוי דוברים הצליח! להלן הפלט:")
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            print(f"{turn.start:.2f}s - {turn.end:.2f}s | דובר: {speaker}")
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
    except Exception as e:
        print("❌ שגיאה בעת ביצוע זיהוי דוברים. ממשיך ללא זיהוי דוברים.")
        print(e)

    return speaker_segments

def process_audio_file(file_path):
    """
    פונקציה מרכזית לעיבוד קובץ שמע:
      1. בודקת FFmpeg, טוקן וקיום קובץ האודיו.
      2. טוענת את מודל WhisperX, מבצעת תמלול ו-Force Alignment.
      3. מריצה זיהוי דוברים.
      4. אוספת את המילים מהפלט, משייכת דוברים ומסמנת מילים בעייתיות ברמת המילה.
      5. בונה DataFrame ברמת המילה.
    """
    # בדיקת FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        print("✅ FFmpeg זמין!")
    except FileNotFoundError:
        print("❌ FFmpeg לא נמצא במערכת או אינו מוגדר ב-PATH.")
        sys.exit(1)

    # בדיקת טוקן
    if not HUGGING_FACE_TOKEN:
        print("❌ HUGGING_FACE_TOKEN לא מוגדר בסביבת העבודה.")
        sys.exit(1)

    # בדיקה שקובץ האודיו קיים
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        print(f"❌ הקובץ {abs_path} לא קיים!")
        sys.exit(1)
    print(f"🔹 מעבד קובץ אודיו: {abs_path}")

    # טעינת מודל WhisperX ותהליך התמלול וה-Alignment
    print("🔹 טוען מודל WhisperX...")
    whisper_model = whisperx.load_model("large-v2", device=DEVICE, compute_type="float32")
    alignment_model, metadata = whisperx.load_align_model(language_code="he", device=DEVICE)
    audio = whisperx.load_audio(abs_path)

    print("🔹 מבצע תמלול ראשוני...")
    whisper_result = whisper_model.transcribe(audio)
    print("🔹 מסיים תמלול. מזהה שפה:", whisper_result.get("language", "לא ידוע"))

    print("🔹 מבצע Alignment...")
    aligned_result = whisperx.align(
        whisper_result["segments"],
        alignment_model,
        metadata,
        audio,
        DEVICE
    )
    print("✅ Alignment הושלם!")

    # הרצת זיהוי דוברים
    speaker_segments = run_diarization(abs_path)

    # איסוף כל המילים מהפלט (עבור כל segment נאסף השדה "words")
    aligned_words = []
    if "segments" in aligned_result:
        for segment in aligned_result["segments"]:
            if "words" in segment:
                aligned_words.extend(segment["words"])
    else:
        print("❌ לא נמצאו segments בתוצאות Alignment.")

    # שיוך דוברים למילים (אם קיימים)
    if speaker_segments:
        aligned_words = assign_speakers_to_words(aligned_words, speaker_segments)

    # בניית רשימת נתונים לכל מילה
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
            "score": score,  # נשמור את הערך המקורי
            "אחוז ניבוי": score_percent,
            "התחלה": format_time(start_time),
            "סיום": format_time(end_time),
            "משך": round(end_time - start_time, 2),
            "דובר": speaker
        })

    df_words = pd.DataFrame(data)
    df_words.sort_values(by=["התחלה"], inplace=True)

    # חישוב סף דינמי עבור כל המילים (global dynamic threshold)
    df_words["Predict_numeric"] = pd.to_numeric(df_words["Predict"], errors="coerce")
    mean_score = df_words["Predict_numeric"].mean()
    std_score = df_words["Predict_numeric"].std()
    dynamic_threshold = mean_score - std_score
    print(f"ממוצע score: {mean_score:.4f}, סטיית תקן: {std_score:.4f}, סף דינמי: {dynamic_threshold:.4f}")

    # עדכון עמודת "בעייתית" עבור המילים (global)
    df_words["בעייתית"] = df_words["Predict_numeric"].apply(lambda x: True if pd.notnull(x) and x < dynamic_threshold else False)

    print("✅ עיבוד ברמת המילה הסתיים בהצלחה!")
    return df_words, aligned_result

def merge_word_and_segment_data(aligned_result):
    """
    ממזג נתוני Alignment כך שכל שורה ב-DataFrame מייצגת משפט/פסקה.
    אם אין תוצאות, מחזיר DataFrame עם עמודות ריקות כדי למנוע קריסה.
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
                                     if isinstance(w.get("score"), (int, float)) and seg_threshold is not None and w.get("score") < seg_threshold]
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
    audio_file = "audio_files/call_water_fixed_first60s.wav"
    print(f"🔹 מעבד את הקובץ: {audio_file}")

    df_words, aligned_result = process_audio_file(audio_file)
    print("\n🔎 בדיקת נתונים ראשונים עבור מילים (head):")
    print(df_words.head(20).to_string(index=False))

    df_sentences = merge_word_and_segment_data(aligned_result)
    print("\n🔎 בדיקת נתונים ראשונים עבור משפטים (head):")
    print(df_sentences.head(20).to_string(index=False))

    out_csv_words = "output/full_transcription_words.csv"
    df_words.to_csv(out_csv_words, index=False, encoding="utf-8-sig")
    print(f"✅ שמירה ל-CSV עבור מילים בוצעה: '{out_csv_words}'")

    out_csv_sentences = "output/full_transcription_sentences.csv"
    df_sentences.to_csv(out_csv_sentences, index=False, encoding="utf-8-sig")
    print(f"✅ שמירה ל-CSV עבור משפטים בוצעה: '{out_csv_sentences}'")


### CHAT GPT IMPROVMENT
    print (" ****************          ### CHAT GPT IMPROVMENT")
   # קריאה של הקובץ "full_transcription_sentences.csv" והכנת DataFrame
    df = df_sentences
    transcript_df = df[['משפט', 'דובר']].copy()

    # הפקת כותרת, תקציר ורשימת דוברים יחד עם תמלול מעודכן
    title, summary, speakers, updated_transcript_df = generate_title_summary_and_speakers(transcript_df, audio_file)
    print("\n--- תוצאות הפקת כותרת, תקציר ורשימת דוברים ---")
    print("כותרת:", title)
    print("סיכום:", summary)
    print("רשימת דוברים:", speakers)
   # print("\n--- תמלול השיחה המעודכן ---")
   # print(updated_transcript_df.head(20).to_string(index=False))

    # תיקון התמלול בחלוקה לבאצ'ים (עד 10 משפטים בכל פעם)
    print("# תיקון התמלול בחלוקה לבאצ'ים (עד 10 משפטים בכל פעם)************************    ")
    result = correct_transcription_and_summary(updated_transcript_df, title, summary, speakers, audio_file)
    print("\n--- תוצאות תיקון התמלול ---")
    print(result)

    print("# הדפסת הרשומות המסוננות ששונו ----------------------------")
    # סינון הרשומות בהן ChangeAvailable הוא True
    filtered_records = [record for record in result['transcript'] if record.get("ChangeAvailable") is True]
    for record in filtered_records:
        print(record)
