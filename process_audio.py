import sys
# מרשה עד 20000 קריאות רקורסיביות
sys.setrecursionlimit(20000)

import time
import pandas as pd
from TranscribeAI import process_audio_file, run_diarization, merge_word_and_segment_data
from chatGpt_Improvement import generate_title_summary_and_speakers, correct_transcription_and_summary

class AudioProcessor:
    """מחלקה שמבצעת עיבוד קובץ אודיו ומעדכנת סטטוס בזמן אמת"""

    def __init__(self, processing_id, file_path, status_dict):
        self.processing_id = processing_id
        self.file_path = file_path
        self.status_dict = status_dict

    def update_status(self, status):
        """ עדכון הסטטוס במערכת """
        self.status_dict[self.processing_id] = {"status": status}

    def process(self):
        """ תהליך מלא של תמלול, זיהוי דוברים ותיקון תמלול """
        print(f"[%s] התחלת עיבוד הקובץ: {self.file_path}" % self.processing_id)

        try:
            self.update_status("🔄 מעבד את הקובץ...")
            df_words, aligned_result = process_audio_file(self.file_path)

            self.update_status("🔎 מנתח את המשפטים והמילים...")
            print("🔎 מנתח את המשפטים והמילים...")
            df_sentences = merge_word_and_segment_data(aligned_result)

            # 🛠️ נוודא התאמה לשמות עמודות גם אם בעברית
            if "speech" not in df_sentences.columns:
                if "משפט" in df_sentences.columns:
                    df_sentences.rename(columns={"משפט": "speech"}, inplace=True)
                    print("ℹ️ המרה: 'משפט' -> 'speech'")
                else:
                    print("❌ שגיאה: עמודת 'speech' חסרה ב- df_sentences.")
                    self.update_status("❌ שגיאה: לא ניתן להמשיך, עמודת 'speech' חסרה.")
                    return

            if "speaker" not in df_sentences.columns:
                if "דובר" in df_sentences.columns:
                    df_sentences.rename(columns={"דובר": "speaker"}, inplace=True)
                    print("ℹ️ המרה: 'דובר' -> 'speaker'")
                else:
                    print("❌ שגיאה: עמודת 'speaker' חסרה ב- df_sentences.")
                    self.update_status("❌ שגיאה: לא ניתן להמשיך, עמודת 'speaker' חסרה.")
                    return

            self.update_status("🗣️ מזהה דוברים...")
            print("🗣️ מזהה דוברים...")
            transcript_df = df_sentences[["speech", "speaker"]].copy()

            print(transcript_df.head(20).to_string(index=False))
            self.update_status("✍️ מחולל כותרת, תיאור וזיהוי דוברים חכם...")
            title, summary, speakers, updated_transcript_df = generate_title_summary_and_speakers(transcript_df, self.processing_id)

            #print("_____________updated_transcr#ipt_df")
            #print(updated_transcript_df.head(20).to_string(index=False))

            if "speech" not in updated_transcript_df.columns:
                updated_transcript_df["speech"] = updated_transcript_df.get("משפט", "")




            print("==== title: " + title)
            print("==== summary: " + summary)
            print("==== speakers: " + str(speakers))
            print("==== updated_transcript_df: " +  updated_transcript_df.head(20).to_string(index=False))

            # result =updated_transcript_df
            self.update_status("✍️ ניסיון לתיקון חכם של התמלול...")
            print("✍️ ניסיון לתיקון חכם של התמלול...")
            result = correct_transcription_and_summary(updated_transcript_df, title, summary, speakers)



            print("               TRY TO PRINTTTTTTTTTT             ")
            filtered_result = [item for item in result["transcript"] if isinstance(item, dict)]
            df_transcript = pd.DataFrame(filtered_result)
            print("==== result FINAL: " + df_transcript.head(20).to_string(index=False))



            out_csv = "output/resultGPT.csv"
            if isinstance(df_transcript, pd.DataFrame):
                df_transcript.to_csv(out_csv, index=False, encoding="utf-8-sig")
                print("✅ שמירה ל-CSV df_transcript")
            else:
                print("❌ לא ניתן לשמור ל-CSV: result אינו DataFrame")

            # הכנת תמלול שיחה מקורי
            if isinstance(df_transcript, pd.DataFrame):
                transcript = "\n".join(df_transcript["speech"].dropna().astype(str).tolist())
            else:
                transcript = "❌ אין נתוני תמלול זמינים"
            print("^^^^^^^^^^^^^^^^^^^^^^  transcript orginal: \n" + transcript)

            # הכנת תמלול שיחה מקורי
            if isinstance(df_transcript, pd.DataFrame):
                correctedTranscript = "\n".join(df_transcript["NewSpeech"].dropna().astype(str).tolist())
            else:
                correctedTranscript = "❌ אין נתוני תמלול זמינים"
            print("^^^^^^^^^^^^^^^^^^^^^^^  corrected Transcript \n" + correctedTranscript)

            # נניח ש-df_transcript הוא DataFrame עם העמודות 'speaker', 'speech' ו-'NewSpeech'
            html_transcript = ""
            for index, row in df_transcript.iterrows():
                speaker = row.get("speaker", "")
                speech = row.get("speech", "")
                # נבנה פסקה בה שם הדובר מודגש בכחול
                html_transcript += f'<span style="color:blue; font-weight:bold;">{speaker}:</span> {speech}\n'

            html_corrected = ""
            for index, row in df_transcript.iterrows():
                speaker = row.get("speaker", "")
                new_speech = row.get("NewSpeech", "")
                # שם הדובר מודגש בכחול, והטקסט המתוקן יוצג בצבע ירוק
                html_corrected += f'<span style="color:blue; font-weight:bold;">{speaker}:</span> <span style="color:green;">{new_speech}</span>\n'


            self.status_dict[self.processing_id] = {
                "status": "✅ הושלם!",
                "result": {
                    "title": title,
                    "summary": summary,
                    "speakers": speakers,
                    "transcript": html_transcript,
                    "correctedTranscript": html_corrected
                }
            }

        except Exception as e:
            self.update_status(f"❌ שגיאה בעיבוד: {str(e)}")
