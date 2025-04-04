# שימוש ב- Python 3.10 כבסיס
FROM python:3.10

# התקנת FFmpeg
RUN apt update && apt install -y ffmpeg

# יצירת תיקיות נחוצות
WORKDIR /app
COPY . /app

# התקנת תלויות
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# הרצת Flask
CMD ["python", "app.py"]
ד