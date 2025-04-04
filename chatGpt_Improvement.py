import os
import openai
import json
import re
import pandas as pd

# הגדרת מפתח ה-API של OpenAI (ודא שהמשתנה OPENAI_API_KEY מוגדר בסביבת העבודה שלך)
openai.api_key = os.getenv("OPENAI_API_KEY")

# יצירת מופע לקוח של OpenAI
client = openai.OpenAI()

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
    try:
        return json.loads(text)
    except Exception as e:
        print("❌ ניסיון ראשוני לפענוח JSON נכשל:", e)
        print("טקסט גולמי:", repr(text))

        # נסה לחלץ בלוק JSON חוקי עם REGEX
        match = re.search(r'{.*}', text, re.DOTALL)
        if match:
            candidate = match.group()
            try:
                return json.loads(candidate)
            except Exception as inner_e:
                print("❌ גם ניסיון חילוץ JSON נכשל:", inner_e)
        raise ValueError("❌ לא ניתן לפענח JSON מהטקסט.")


def generate_title_summary_and_speakers(transcript_df, audio_file):
    """
    יוצר כותרת, תקציר ורשימת דוברים מתמלול השיחה.
    """
    if transcript_df.empty or not all(col in transcript_df.columns for col in ["speech", "speaker"]):
        print("❌ שגיאה: transcript_df חסר עמודות נדרשות!")
        return "", "", [], transcript_df.copy()

    transcript_text = "\n".join(f"{row['speaker']}: {row['speech']}" for _, row in transcript_df.iterrows())
    print("$$$$$$$$$$$$$$$$ transcript_text: " + transcript_text)
    try:
        prompt = PROMPTS.get("title_summary_speakers", "").format(transcript_text=transcript_text)
        print("############### prompt: " + prompt)
    except Exception as e:
        print("❌ שגיאה בפענוח prompt:", e)

    try:

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "אתה מומחה לניתוח שיחות והפקת כותרות."},
                      {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )


        answer = response.choices[0].message.content

        print("############### answer: " +answer)

        try:
            result = safe_json_loads(answer)
        except Exception as e:
            print("❌ שגיאה בפענוח JSON:", e)
            result = {"title": "", "summary": "", "speakers": [], "transcript": []}

       #print("############### result: " +result)
        title = result.get("title", "")
        summary = result.get("summary", "")
        speakers = result.get("speakers", [])
        transcript_list = result.get("transcript", [])

        print("==== title: " + title)
        print("==== summary: " + summary)
        print("==== speakers: " + str(speakers))
        print("\n==== transcript_list ====")
        for item in transcript_list:
            print(f'{item.get("speaker", "")}: {item.get("speech", "")}')

        updated_df = pd.DataFrame(transcript_list) if transcript_list else transcript_df.copy()
        print("====  ====  updated_df " + updated_df.head(20).to_string(index=False))
    except Exception as e:
        print(f"❌ קרתה שגיאה בעת עיבוד התשובה מ-ChatGPT: {e}")
        title, summary, speakers, updated_df = "", "", [], transcript_df.copy()

    return title, summary, speakers, updated_df


def correct_transcription_and_summary(transcript_df, title, summary, speakers, batch_size=10):
    # בדיקה אם transcript_df ריק או חסרים בו העמודות "speech" ו-"speaker"
    if transcript_df.empty or not all(col in transcript_df.columns for col in ["speech", "speaker"]):
        print("❌ שגיאה: transcript_df חסר עמודות נדרשות!")
        return {"transcript": []}

    # יצירת טקסט תמלול לצורך הדפסה או שימוש בהמשך
    transcript_text = "\n".join(f"{row['speaker']}: {row['speech']}" for _, row in transcript_df.iterrows())
    print("$$$$$$$$$$$$$$$$ transcript_text: " + transcript_text)

    corrected_transcript = []
    try:
        transcript_records = transcript_df.to_dict(orient="records")
        batches = [transcript_records[i:i + batch_size] for i in range(0, len(transcript_records), batch_size)]
        for batch in batches:
            batch_text = "\n".join(f"{record.get('speaker', '')}: {record.get('speech', '')}" for record in batch)
            prompt = PROMPTS.get("transcription_correction", "").format(
                title=title,
                summary=summary,
                speakers=", ".join(speakers),
                batch_text=batch_text
            )
            print("############### prompt: " + prompt)
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "אתה מומחה לתמלולים ותיקון טקסטים."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000,
                )
                answer = response.choices[0].message.content
                print("############### answer: " + answer)
                try:
                    result = safe_json_loads(answer)
                    print("############### result: " + str(result))
                    corrected_transcript.extend(result.get("transcript", []))
                except Exception as e:
                    print("❌ שגיאה בפענוח JSON:", e)
                finally:
                    corrected_transcript.append("")  # הוספת שורה ריקה
            except Exception as e:
                print("❌ שגיאה בקריאת CHAT GPT:", e)
    except Exception as e:
        print("❌ שגיאה בתמלול חכם:", e)

    print("!@!@!@!@!@   corrected_transcript")
    print(corrected_transcript)
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

    #result = correct_transcription_and_summary(updated_transcript_df, title, summary, speakers, audio_file)
    #print("\n--- תוצאות תיקון התמלול ---")
    #print(result)
    result =updated_transcript_df

    # הדפסת משפטים ששונו
    filtered_records = [record for record in result['transcript'] if record.get("ChangeAvailable") is True]
    for record in filtered_records:
        print(record)
