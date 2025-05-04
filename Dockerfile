FROM python:3.10-slim

# מנקים cache, מעדכנים pip ומתקינים תלויות
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# העברת קוד האפליקציה
COPY . .

# משתני סביבה (לדוגמה)
ENV PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

# נקודת הכניסה – תואם ל‑app.yaml
CMD ["gunicorn", "-k", "gevent", "-b", ":8080", "--timeout=900", "--log-level", "debug", "app:app"]
