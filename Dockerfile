FROM python:3.10-slim

# Install dependensi OS untuk OpenCV, MediaPipe, dan MySQL compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Menjalankan aplikasi dengan Gunicorn production worker
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]