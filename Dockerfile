FROM python:3.9-slim

# הגדרת מצב noninteractive למניעת בקשות קלט במהלך הבנייה
ENV DEBIAN_FRONTEND=noninteractive

# לכפות שימוש ב־IPv4 עבור apt-get (עוזר בבעיות DNS)
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4

# החלפת כתובות המקורות ל־HTTPS במידה והקובץ קיים
RUN if [ -f /etc/apt/sources.list ]; then \
      sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list; \
    fi && \
    if [ -d /etc/apt/sources.list.d ]; then \
      sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/* || true; \
    fi

# עדכון המאגרים והתקנת FFmpeg, וניקוי קבצי המטמון
RUN apt-get update --fix-missing && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# הגדרת תיקיית העבודה לאפליקציה
WORKDIR /app

RUN mkdir -p uploads

# העתקת קבצי הדרישות (ודאו שקבצי requirements.txt ו־constraints.txt קיימים, גם אם constraints.txt הוא קובץ ריק)
COPY requirements.txt ./
COPY constraints.txt ./

# העתקת תיקיית הקבצים הסטטיים – ודאו שהתיקייה קיימת ושמה תואם (לדוגמה "static" עם אותיות קטנות)
RUN mkdir -p STATIC
COPY STATIC/prompts.json STATIC/

RUN mkdir -p uploads

# [אופציונלי] העתקת תיקיית המודל – הסר או הגב שורה זו אם אין תיקייה בשם "model"
# COPY model /root/.cache/huggingface/hub/models--Systran--faster-whisper-large-v2/snapshots/f0fe81560cb8b68660e564f55dd99207059c092e/

# התקנת ספריות הפייתון מהתלויות, עם timeout מוגבר (300 שניות)
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt
RUN pip install huggingface_hub[hf_xet]
RUN pip install hf_xet

# העתקת שאר קבצי הפרויקט לתיקיית העבודה
COPY . .

# הפעלת האפליקציה עם Gunicorn עם timeout מוגבר (600 שניות)
CMD ["gunicorn", "-k", "gevent", "-b", ":8080", "--timeout", "600", "app:app"]
