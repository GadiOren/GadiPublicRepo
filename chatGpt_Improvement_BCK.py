import os
import openai
import json
import re
import pandas as pd

# הגדרת מפתח ה-API של OpenAI (ודא שהמשתנה OPENAI_API_KEY מוגדר בסביבת העבודה שלך)
openai.api_key = os.getenv("OPENAI_API_KEY")

# נתיב לקובץ הפרומפטים בספריית STATIC
PROMPTS_FILE_PATH = os.path.join("STATIC", "prompts.json")


def load_prompts():
    """ טוען את הפרומפטים מקובץ JSON """
    try:
        with open(PROMPTS_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ שגיאה בטעינת קובץ הפרומפטים ({PROMPTS_FILE_PATH}): {e}")
        return {}


PROMPTS = load_prompts()


def safe_json_loads(text):
    """ מנסה לפענח JSON תוך טיפול בבעיות נפוצות """
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
        except Exception:
            match = re.search(r'\{.*\}', candidate, re.DOTALL)
            if match:
                return json.loads(match.group())
        raise ValueError("לא ניתן לתקן את JSON מהטקסט.")


def generate_title_summary_and_speakers(transcript_df, audio_file):
    """
    יוצר כותרת, תקציר ורשימת דוברים מתמלול השיחה.
    """
    transcript_text = "\n".join(f"{row['דובר']}: {row['משפט']}" for _, row in transcript_df.iterrows())

    prompt = PROMPTS.get("title_summary_speakers", "").format(transcript_text=transcript_text)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "אתה מומחה לניתוח שיחות והפקת כותרות."},
                      {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
        answer = response["choices"][0]["message"]["content"].strip()
        result = safe_json_loads(answer)

        title = result.get("title", "")
        summary = result.get("summary", "")
        speakers = result.get("speakers", [])
        transcript_list = result.get("transcript", [])

        updated_df = pd.DataFrame(transcript_list) if transcript_list else transcript_df.copy()
    except Exception as e:
        print("❌ קרתה שגיאה בעת עיבוד התשובה מ-ChatGPT:", e)
        title, summary, speakers, updated_df = "", "", [], transcript_df.copy()

    return title, summary, speakers, updated_df


def correct_transcription_and_summary(updated_transcript_df, title, summary, speakers, audio_file, batch_size=10):
    """
    מתקן את תמלול השיחה באמצעות ChatGPT בחלוקה לבאצ'ים של עד 10 משפטים.
    """
    transcript_records = updated_transcript_df.to_dict(orient="records")
    batches = [transcript_records[i:i + batch_size] for i in range(0, len(transcript_records), batch_size)]
    corrected_transcript = []

    for batch in batches:
        batch_text = "\n".join(f"{record.get('speaker', '')}: {record.get('speech', '')}" for record in batch)

        prompt = PROMPTS.get("transcription_correction", "").format(
            title=title, summary=summary, speakers=", ".join(speakers), batch_text=batch_text
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "אתה מומחה לתמלולים ותיקון טקסטים."},
                          {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
            )
            answer = response["choices"][0]["message"]["content"].strip()
            result = safe_json_loads(answer)
            corrected_transcript.extend(result.get("transcript", []))
        except Exception as e:
            print("❌ קרתה שגיאה בעת עיבוד חלק מהתמלול:", e)

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

    result = correct_transcription_and_summary(updated_transcript_df, title, summary, speakers, audio_file)
    print("\n--- תוצאות תיקון התמלול ---")
    print(result)

    # הדפסת משפטים ששונו
    filtered_records = [record for record in result['transcript'] if record.get("ChangeAvailable") is True]
    for record in filtered_records:
        print(record)
