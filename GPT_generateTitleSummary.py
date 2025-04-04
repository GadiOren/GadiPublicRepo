import os
import openai
import json
import pandas as pd

# הגדרת מפתח ה-API של OpenAI (ודא שהמשתנה OPENAI_API_KEY מוגדר בסביבת העבודה שלך)
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_json_string(text):
    """
    פונקציה שמנסה לחלץ את תת-המחרוזת שמכילה JSON תקין מתוך טקסט.
    במידה ונמצא סימן '{' בסוף סימן '}', היא מחזירה את החלק שביניהם.
    """
    first = text.find('{')
    last = text.rfind('}')
    if first != -1 and last != -1:
        return text[first:last+1]
    return text

def generate_title_summary_and_speakers(transcript_df, audio_file):
    """
    פונקציה שמקבלת DataFrame עם תמלול השיחה (עמודות "משפט" ו-"דובר"),
    מאחדת את התמלול, בונה prompt עבור ChatGPT, ומבקשת ממנו:
      - לסכם את השיחה ולהפיק כותרת/נושא,
      - לחלץ רשימת דוברים,
      - ולעדכן את התמלול כך ששמות הדוברים יהיו מתוקנים (במידה והתרחשו טעויות בזיהוי).

    הפלט צריך להיות בפורמט JSON עם ארבעה מפתחות:
      'title'      – כותרת השיחה,
      'summary'    – תקציר השיחה,
      'speakers'   – מערך עם שמות הדוברים,
      'transcript' – תמלול השיחה המעודכן (רשימה של רשומות, כל רשומה עם 'משפט' ו-'דובר').
    """
    # איחוד תמלול – כל שורה בפורמט "דובר: משפט"
    transcript_text = "\n".join(f"{row['דובר']}: {row['משפט']}" for _, row in transcript_df.iterrows())

    # בניית prompt עם כל הפרטים
    prompt = (
        "אתה מומחה לתמלולים ולניתוח שיחות טלפוניות. "
        "להלן תמלול של שיחה עם דוברים שונים שהוקלטה ותומללה באופן אוטומטי. "
        "אנא סכם את השיחה בצורה מקצועית, הפק כותרת/נושא שמשקף את הנושאים המרכזיים, "
        "וחלץ רשימת דוברים. בנוסף, עדכן את תמלול השיחה כך שכל שורה תציג את שם הדובר המעודכן במידה והתרחשה טעות בזיהוי. "
        "הפלט שלך צריך להיות בפורמט JSON עם ארבעה מפתחות: 'title', 'summary', 'speakers', 'transcript'.\n\n"
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

        # חילוץ תת-מחרוזת JSON תקינה מהתשובה
        answer_json = extract_json_string(answer)
        result = json.loads(answer_json)
        title = result.get("title", "")
        summary = result.get("summary", "")
        speakers = result.get("speakers", [])
        transcript_list = result.get("transcript", [])

        # אם מתקבל תמלול מעודכן, נבנה ממנו DataFrame, אחרת נשתמש במקורי
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


def correct_transcription_and_summary(updated_transcript_df, title, summary, speakers, audio_file):
    """
    פונקציה שמביאה לתיקון טקסט השיחה בצורה מקצועית.
    היא שולחת את המידע הבא לצ'אט GPT:
      - כותרת, סיכום, רשימת דוברים,
      - תמלול השיחה לפני התיקון (כפי שקיבלנו מהפונקציה הקודמת)
    ומבקשת ממנו לבצע תיקונים בטקסט בצורה מקצועית ורק במידה והוא בטוח.

    הפלט צריך להיות בפורמט JSON עם המפתחות:
      'audio_file'         – שם הקובץ,
      'title'              – כותרת מתוקנת,
      'summary'            – סיכום מתוקן,
      'speakers'           – רשימת דוברים לאחר תיקון,
      'transcript_after'   – תמלול שיחה חדש לאחר תיקון,
      'transcript_before'  – תמלול שיחה לפני תיקון.
    """
    # המרת DataFrame של התמלול לפני תיקון למחרוזת (טקסט)
    transcript_before = updated_transcript_df.to_string(index=False)

    prompt = (
        "אתה מומחה לתיקון תמלולים, בעל עין מקצועית וערנות לשימור דיוק הטקסט. "
        "להלן מידע שמתקבל מתמלול שיחה: כותרת, סיכום, רשימת דוברים ותמלול שיחה לפני תיקון. "
        "אנא בצע תיקון לטקסט בצורה מקצועית ומהימנה, תוך שמירה על שמות הדוברים. "
        "בצע תיקון רק אם אתה בטוח שזו המילה הנכונה. "
        "אל תערוך שינויים במידה ואינך בטוח.\n\n"
        f"שם הקובץ: {audio_file}\n"
        f"כותרת: {title}\n"
        f"סיכום: {summary}\n"
        f"רשימת דוברים: {', '.join(speakers)}\n\n"
        "תמלול שיחה לפני תיקון:\n" + transcript_before + "\n\n"
        "אנא הפק את הפלט בפורמט JSON עם המפתחות הבאים: "
        "'audio_file', 'title', 'summary', 'speakers', 'transcript_after', 'transcript_before'."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "אתה מומחה לתיקון תמלולים."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        answer = response["choices"][0]["message"]["content"].strip()
        print("✅ קיבלנו תשובה מ-ChatGPT (תיקון תמלול):")
        print(answer)

        # חילוץ תת-מחרוזת JSON תקינה מהתשובה
        answer_json = extract_json_string(answer)
        result = json.loads(answer_json)
        corrected_title = result.get("title", "")
        corrected_summary = result.get("summary", "")
        corrected_speakers = result.get("speakers", [])
        transcript_after = result.get("transcript_after", "")
        transcript_before = result.get("transcript_before", transcript_before)
    except Exception as e:
        print("❌ קרתה שגיאה בעת עיבוד התשובה מ-ChatGPT (תיקון):", e)
        corrected_title = ""
        corrected_summary = ""
        corrected_speakers = []
        transcript_after = transcript_before
        transcript_before = transcript_before

    return {
        "audio_file": audio_file,
        "title": corrected_title,
        "summary": corrected_summary,
        "speakers": corrected_speakers,
        "transcript_after": transcript_after,
        "transcript_before": transcript_before
    }


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
    print("שם הקובץ:", result["audio_file"])
    print("כותרת מתוקנת:", result["title"])
    print("סיכום מתוקן:", result["summary"])
    print("רשימת דוברים לאחר תיקון:", result["speakers"])
    print("\n--- תמלול שיחה חדש לאחר תיקון ---")
    print(result["transcript_after"])
    print("\n--- תמלול שיחה לפני תיקון ---")
    print(result["transcript_before"])
