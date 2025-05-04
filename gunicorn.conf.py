# Bind to all network interfaces on port 8080
bind = "0.0.0.0:8080"
# Number of worker processes (tune as needed)
workers = 2
# Number of threads per worker
threads = 4
# Adjust timeout to allow long-running transcription
timeout = 600
# Preload app code (warm-up at startup)
preload = True