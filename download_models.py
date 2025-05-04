# download_models.py
from huggingface_hub import snapshot_download
import os

# allow override via ARG
token = os.getenv("HUGGING_FACE_TOKEN", None)

# 1️⃣ WhisperX (large-v2)
snapshot_download(
    repo_id="Systran/faster-whisper-large-v2",
    cache_dir="/root/.cache/huggingface",
    use_auth_token=token,
)

# 2️⃣ Alignment model
# (you can add more here if needed)
snapshot_download(
    repo_id="openai/whisper-large-v2",
    cache_dir="/root/.cache/huggingface",
    use_auth_token=token,
)

print("✅ Models downloaded into /root/.cache/huggingface")
