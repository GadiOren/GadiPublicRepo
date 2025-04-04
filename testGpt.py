import os
import openai
import json
import re
import pandas as pd

# הגדרת מפתח ה-API של OpenAI (ודא שהמשתנה OPENAI_API_KEY מוגדר בסביבת העבודה שלך)
openai.api_key = os.getenv("OPENAI_API_KEY")


def safe_json_loads(text):
    """
    מנסה לפרש טקסט כ-JSON. במקרה של שגיאה (למשל, JSON מקוצר או חסר סוגריים),
    הפונקציה מנסה לתקן את המחרוזת:
      1. מוצאת את תחילת ה-JSON (תו '{').
      2. בודקת אם המחרוזת מסתיימת בסוגר סוגר – ואם לא, מוסיפה אותו.
      3. במידת הצורך, מנסה לחתוך חלקים לא שלמים מהסוף.
    """
    try:
        return json.loads(text)
    except Exception as e:
        print("❌ ניסיון ראשוני לפענוח JSON נכשל:", e)
        first = text.find('{')
        if first == -1:
            raise ValueError("לא נמצא התחלת JSON בטקסט")
        candidate = text[first:]
        if not candidate.rstrip().endswith('}'):
            candidate += "}"
        try:
            return json.loads(candidate)
        except Exception as e2:
            print("❌ ניסיון נוסף עם השלמת סוגריים נכשל:", e2)
            last_quote = candidate.rfind('"')
            if last_quote != -1:
                candidate = candidate[:last_quote + 1] + "}"
                try:
                    return json.loads(candidate)
                except Exception as e3:
                    print("❌ ניסיון לתיקון JSON על ידי חיתוך הטקסט נכשל:", e3)
            match = re.search(r'\{.*\}', candidate, re.DOTALL)
            if match:
                candidate = match.group()
                try:
                    return json.loads(candidate)
                except Exception as e4:
                    print("❌ גם ניסיון באמצעות ביטוי רגולרי נכשל:", e4)
        raise ValueError("לא ניתן לתקן את JSON מהטקסט.")


def generate_title_summary_and_speakers(transcript_df, audio_file):
    """
    פונקציה המקבלת DataFrame עם תמלול (עמודות "משפט" ו-"דובר"),
    מאחדת את התמלול, בונה prompt עבור ChatGPT, ומבקשת ממנו:
      - לסכם את השיחה ולהפיק כותרת/נושא,
      - לחלץ רשימת דוברים,
      - ולעדכן את תמלול השיחה כך ששמות הדוברים יהיו מתוקנים.

    הפלט הוא JSON עם המפתחות:
      'title', 'summary', 'speakers', 'transcript'
    כאשר 'transcript' הוא מערך של רשומות עם השדות:
      'speaker' – מי דיבר,
      'speech' – הטקסט המקורי.
    """
    transcript_text = "\n".join(f"{row['דובר']}: {row['משפט']}" for _, row in transcript_df.iterrows())

    prompt = (
            "אתה מומחה לתמלולים ולניתוח שיחות טלפוניות. "
            "להלן תמלול של שיחה עם דוברים שונים שהוקלטה ותומללה באופן אוטומטי. "
            "אנא סכם את השיחה בצורה מקצועית, הפק כותרת/נושא שמשקף את הנושאים המרכזיים, "
            "וחלץ רשימת דוברים. בנוסף, עדכן את תמלול השיחה כך שכל שורה תציג את שם הדובר המעודכן במידה והתרחשה טעות בזיהוי. "
            "הפלט שלך צריך להיות אך ורק JSON תקין עם המפתחות: 'title', 'summary', 'speakers', 'transcript'.\n\n"
            "תמלול השיחה:\n" + transcript_text
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "אתה מומחה לניתוח שיחות והפקת כותרות."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        answer = response["choices"][0]["message"]["content"].strip()
        print("✅ קיבלנו תשובה מ-ChatGPT (הפקת כותרת, תקציר ורשימת דוברים):")
        print(answer)

        result = safe_json_loads(answer)
        title = result.get("title", "")
        summary = result.get("summary", "")
        speakers = result.get("speakers", [])
        transcript_list = result.get("transcript", [])

        # אם נמצא מפתח "utterance" במקום "speech" – נתקן
        for rec in transcript_list:
            if "utterance" in rec and "speech" not in rec:
                rec["speech"] = rec.pop("utterance")
        if transcript_list:
            updated_df = pd.DataFrame(transcript_list)
        else:
            updated_df = transcript_df.copy()
    except Exception as e:
        print("❌ קרתה שגיאה בעת עיבוד התשובה מ-ChatGPT:", e)
        title = ""
        summary = ""
        speakers = []
        updated_df = transcript_df.copy()

    return title, summary, speakers, updated_df


def correct_transcription_and_summary(updated_transcript_df, title, summary, speakers, audio_file, batch_size=10):
    """
    פונקציה לתיקון תמלול השיחה.
    מחלקת את התמלול לחלקים (עד 10 משפטים בכל בקשה) ושולחת כל חלק לבקשת תיקון ל-ChatGPT.
    ההנחיות:
      - עבור כל משפט, בצע תיקון לטקסט המקורי רק אם אתה בטוח מאוד שהתיקון משפר את המשמעות או הבהירות.
      - אם אין תיקון ברור, השאר את הטקסט כפי שהוא.
      - עבור כל משפט, הפלט חייב להיות אך ורק JSON, ללא טקסט נוסף.
      - הפלט צריך להיות אובייקט JSON עם מפתח "transcript" המכיל מערך של אובייקטים.
        כל אובייקט מייצג משפט ומכיל EXACTLY את השדות:
           'speaker'        – מי דיבר,
           'speech'         – הטקסט המקורי,
           'NewSpeech'      – הטקסט המתוקן (אם אין תיקון, זהה ל-'speech'),
           'ChangeAvailable'– TRUE אם הייתה הצעת תיקון מוצדקת, אחרת FALSE.

    ההקשר (כותרת, סיכום, רשימת דוברים) מועבר לצורך שיפור הטקסט.
    במידה והתמלול ארוך, הפונקציה מחלקת אותו לבקשות של עד 10 משפטים בכל פעם.

    הפונקציה מחזירה מילון עם המפתח "transcript" המכיל את רשימת המשפטים לאחר התיקון.
    """
    transcript_records = updated_transcript_df.to_dict(orient="records")
    batches = [transcript_records[i:i + batch_size] for i in range(0, len(transcript_records), batch_size)]
    corrected_transcript = []

    for batch in batches:
        batch_text = "\n".join(f"{record.get('speaker', '')}: {record.get('speech', '')}" for record in batch)
        prompt = (
            "אתה מומחה לתמלולים ותיקון טקסטים. "
            "עבור כל משפט בתמלול הבא, בצע תיקון לטקסט המקורי רק אם אתה בטוח מאוד שהתיקון משפר את המשמעות או הבהירות. "
            "אם אין תיקון ברור, השאר את הטקסט כפי שהוא.\n\n"
            "עליך להוציא אך ורק JSON, ללא שום טקסט נוסף. הפלט צריך להיות אובייקט JSON עם מפתח 'transcript' המכיל מערך של אובייקטים. "
            "כל אובייקט מייצג משפט ומכיל EXACTLY את השדות הבאים:\n"
            "  'speaker'        – שם הדובר,\n"
            "  'speech'         – הטקסט המקורי,\n"
            "  'NewSpeech'      – הטקסט המתוקן (אם אין תיקון, זהה ל-'speech'),\n"
            "  'ChangeAvailable'– TRUE אם הייתה הצעת תיקון מוצדקת, אחרת FALSE.\n\n"
            "להלן הקשר לשיפור התמלול:\n"
            f"כותרת: {title}\n"
            f"סיכום: {summary}\n"
            f"רשימת דוברים: {', '.join(speakers)}\n\n"
            "תמלול לשיפור:\n"
            f"{batch_text}\n"
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "אתה מומחה לתמלולים ותיקון טקסטים."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            answer = response["choices"][0]["message"]["content"].strip()
            print("✅ קיבלנו תשובה עבור חלק:\n", answer)
            result = safe_json_loads(answer)
            batch_transcript = result.get("transcript", [])
            corrected_transcript.extend(batch_transcript)
        except Exception as e:
            print("❌ קרתה שגיאה בעת עיבוד חלק מהתמלול:", e)
            continue

    return {"transcript": corrected_transcript}


if __name__ == "__main__":
    df = pd.read_csv("full_transcription_sentences.csv")
    transcript_df = df[['משפט', 'דובר']].copy()
    audio_file = "example_audio.wav"

    title, summary, speakers, updated_transcript_df = generate_title_summary_and_speakers(transcript_df, audio_file)
    print("\n--- תוצאות הפקת כותרת, תקציר ורשימת דוברים ---")
    print("כותרת:", title)
    print("סיכום:", summary)
    print("רשימת דוברים:", speakers)
    print("\n--- תמלול השיחה המעודכן ---")
    print(updated_transcript_df.head(20).to_string(index=False))

    result = correct_transcription_and_summary(updated_transcript_df, title, summary, speakers, audio_file)
    print("\n--- תוצאות תיקון התמלול ---")
    print(result)

    print("# הדפסת הרשומות המסוננו ששונו ----------------------------")
    # סינון הרשומות בהן ChangeAvailable הוא True
    filtered_records = [record for record in result['transcript'] if record.get("ChangeAvailable") is True]
    # הדפסת הרשומות המסוננות
    for record in filtered_records:
        print(record)
